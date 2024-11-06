// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 10/2021
#include <algorithm>
#include <set>
#include <vector>

#include "tbb/task_arena.h"
#include "tbb/tbb.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyFastJet.h"

#include "TrackstersPCA.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "fastjet/ClusterSequence.hh"

using namespace ticl;
using namespace fastjet;

template <typename TILES>
PatternRecognitionbyFastJet<TILES>::PatternRecognitionbyFastJet(const edm::ParameterSet &conf,
                                                                edm::ConsumesCollector iC)
    : PatternRecognitionAlgoBaseT<TILES>(conf, iC),
      caloGeomToken_(iC.esConsumes<CaloGeometry, CaloGeometryRecord>()),
      antikt_radius_(conf.getParameter<double>("antikt_radius")),
      minNumLayerCluster_(conf.getParameter<int>("minNumLayerCluster")),
      computeLocalTime_(conf.getParameter<bool>("computeLocalTime")){};

template <typename TILES>
void PatternRecognitionbyFastJet<TILES>::buildJetAndTracksters(std::vector<PseudoJet> &fjInputs,
                                                               std::vector<ticl::Trackster> &result) {
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Basic) {
    edm::LogVerbatim("PatternRecogntionbyFastJet")
        << "Creating FastJet with " << fjInputs.size() << " LayerClusters in input";
  }
  fastjet::ClusterSequence sequence(fjInputs, JetDefinition(antikt_algorithm, antikt_radius_));
  auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets(0));
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Basic) {
    edm::LogVerbatim("PatternRecogntionbyFastJet") << "FastJet produced " << jets.size() << " jets/trackster";
  }

  auto trackster_idx = result.size();
  auto jetsSize = std::count_if(jets.begin(), jets.end(), [this](fastjet::PseudoJet jet) {
    return jet.constituents().size() > static_cast<unsigned int>(minNumLayerCluster_);
  });
  result.resize(trackster_idx + jetsSize);

  for (const auto &pj : jets) {
    if (pj.constituents().size() > static_cast<unsigned int>(minNumLayerCluster_)) {
      for (const auto &component : pj.constituents()) {
        result[trackster_idx].vertices().push_back(component.user_index());
        result[trackster_idx].vertex_multiplicity().push_back(1);
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Basic) {
          edm::LogVerbatim("PatternRecogntionbyFastJet")
              << "Jet has " << pj.constituents().size() << " components that are stored in trackster " << trackster_idx;
        }
      }
      trackster_idx++;
    } else {
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecogntionbyFastJet")
            << "Jet with " << pj.constituents().size() << " constituents discarded since too small wrt "
            << minNumLayerCluster_;
      }
    }
  }
  fjInputs.clear();
}

template <typename TILES>
void PatternRecognitionbyFastJet<TILES>::makeTracksters(
    const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
    std::vector<Trackster> &result,
    std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  // Protect from events with no seeding regions
  if (input.regions.empty())
    return;

  edm::EventSetup const &es = input.es;
  const CaloGeometry &geom = es.getData(caloGeomToken_);
  rhtools_.setGeometry(geom);

  constexpr auto isHFnose = std::is_same<TILES, TICLLayerTilesHFNose>::value;
  constexpr int nEtaBin = TILES::constants_type_t::nEtaBins;
  constexpr int nPhiBin = TILES::constants_type_t::nPhiBins;

  // We need to partition the two sides of the HGCAL detector
  auto lastLayerPerSide = static_cast<unsigned int>(rhtools_.lastLayer(isHFnose)) - 1;
  unsigned int maxLayer = 2 * lastLayerPerSide - 1;
  std::vector<fastjet::PseudoJet> fjInputs;
  fjInputs.clear();
  for (unsigned int currentLayer = 0; currentLayer <= maxLayer; ++currentLayer) {
    if (currentLayer == lastLayerPerSide) {
      buildJetAndTracksters(fjInputs, result);
    }
    const auto &tileOnLayer = input.tiles[currentLayer];
    for (int ieta = 0; ieta <= nEtaBin; ++ieta) {
      auto offset = ieta * nPhiBin;
      if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
        edm::LogVerbatim("PatternRecogntionbyFastJet") << "offset: " << offset;
      }
      for (int iphi = 0; iphi <= nPhiBin; ++iphi) {
        if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
          edm::LogVerbatim("PatternRecogntionbyFastJet") << "iphi: " << iphi;
          edm::LogVerbatim("PatternRecogntionbyFastJet") << "Entries in tileBin: " << tileOnLayer[offset + iphi].size();
        }
        for (auto clusterIdx : tileOnLayer[offset + iphi]) {
          // Skip masked layer clusters
          if (input.mask[clusterIdx] == 0.) {
            if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
              edm::LogVerbatim("PatternRecogntionbyFastJet") << "Skipping masked layerIdx " << clusterIdx;
            }
            continue;
          }
          // Should we correct for the position of the PV?
          auto const &cl = input.layerClusters[clusterIdx];
          math::XYZVector direction(cl.x(), cl.y(), cl.z());
          direction = direction.Unit();
          direction *= cl.energy();
          auto fpj = fastjet::PseudoJet(direction.X(), direction.Y(), direction.Z(), cl.energy());
          fpj.set_user_index(clusterIdx);
          fjInputs.push_back(fpj);
        }  // End of loop on the clusters on currentLayer
      }  // End of loop over phi-bin region
    }  // End of loop over eta-bin region
  }  // End of loop over layers

  // Collect the jet from the other side wrt to the one taken care of inside the main loop above.
  buildJetAndTracksters(fjInputs, result);

  ticl::assignPCAtoTracksters(result,
                              input.layerClusters,
                              input.layerClustersTime,
                              rhtools_.getPositionLayer(rhtools_.lastLayerEE(isHFnose), isHFnose).z(),
                              rhtools_,
                              computeLocalTime_);

  // run energy regression and ID
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Basic) {
    for (auto const &t : result) {
      edm::LogVerbatim("PatternRecogntionbyFastJet") << "Barycenter: " << t.barycenter();
      edm::LogVerbatim("PatternRecogntionbyFastJet") << "LCs: " << t.vertices().size();
      edm::LogVerbatim("PatternRecogntionbyFastJet") << "Energy: " << t.raw_energy();
      edm::LogVerbatim("PatternRecogntionbyFastJet") << "Regressed: " << t.regressed_energy();
    }
  }
}

template <typename TILES>
void PatternRecognitionbyFastJet<TILES>::filter(std::vector<Trackster> &output,
                                                const std::vector<Trackster> &inTracksters,
                                                const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
                                                std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  output = inTracksters;
}

template <typename TILES>
void PatternRecognitionbyFastJet<TILES>::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
  iDesc.add<double>("antikt_radius", 0.09)->setComment("Radius to be used while running the Anti-kt clustering");
  iDesc.add<int>("minNumLayerCluster", 5)->setComment("Not Inclusive");
  iDesc.add<bool>("computeLocalTime", false);
}

template class ticl::PatternRecognitionbyFastJet<TICLLayerTiles>;
