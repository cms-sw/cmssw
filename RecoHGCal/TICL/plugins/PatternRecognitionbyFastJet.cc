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
      eidInputName_(conf.getParameter<std::string>("eid_input_name")),
      eidOutputNameEnergy_(conf.getParameter<std::string>("eid_output_name_energy")),
      eidOutputNameId_(conf.getParameter<std::string>("eid_output_name_id")),
      eidMinClusterEnergy_(conf.getParameter<double>("eid_min_cluster_energy")),
      eidNLayers_(conf.getParameter<int>("eid_n_layers")),
      eidNClusters_(conf.getParameter<int>("eid_n_clusters")){};

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
  auto jetsSize = std::count_if(jets.begin(), jets.end(), [=](fastjet::PseudoJet jet) {
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
      }    // End of loop over phi-bin region
    }      // End of loop over eta-bin region
  }        // End of loop over layers

  // Collect the jet from the other side wrt to the one taken care of inside the main loop above.
  buildJetAndTracksters(fjInputs, result);

  ticl::assignPCAtoTracksters(result,
                              input.layerClusters,
                              input.layerClustersTime,
                              rhtools_.getPositionLayer(rhtools_.lastLayerEE(isHFnose), isHFnose).z());

  // run energy regression and ID
  energyRegressionAndID(input.layerClusters, input.tfSession, result);
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
void PatternRecognitionbyFastJet<TILES>::energyRegressionAndID(const std::vector<reco::CaloCluster> &layerClusters,
                                                               const tensorflow::Session *eidSession,
                                                               std::vector<Trackster> &tracksters) {
  // Energy regression and particle identification strategy:
  //
  // 1. Set default values for regressed energy and particle id for each trackster.
  // 2. Store indices of tracksters whose total sum of cluster energies is above the
  //    eidMinClusterEnergy_ (GeV) threshold. Inference is not applied for soft tracksters.
  // 3. When no trackster passes the selection, return.
  // 4. Create input and output tensors. The batch dimension is determined by the number of
  //    selected tracksters.
  // 5. Fill input tensors with layer cluster features. Per layer, clusters are ordered descending
  //    by energy. Given that tensor data is contiguous in memory, we can use pointer arithmetic to
  //    fill values, even with batching.
  // 6. Zero-fill features for empty clusters in each layer.
  // 7. Batched inference.
  // 8. Assign the regressed energy and id probabilities to each trackster.
  //
  // Indices used throughout this method:
  // i -> batch element / trackster
  // j -> layer
  // k -> cluster
  // l -> feature

  // set default values per trackster, determine if the cluster energy threshold is passed,
  // and store indices of hard tracksters
  std::vector<int> tracksterIndices;
  for (int i = 0; i < static_cast<int>(tracksters.size()); i++) {
    // calculate the cluster energy sum (2)
    // note: after the loop, sumClusterEnergy might be just above the threshold which is enough to
    // decide whether to run inference for the trackster or not
    float sumClusterEnergy = 0.;
    for (const unsigned int &vertex : tracksters[i].vertices()) {
      sumClusterEnergy += static_cast<float>(layerClusters[vertex].energy());
      // there might be many clusters, so try to stop early
      if (sumClusterEnergy >= eidMinClusterEnergy_) {
        // set default values (1)
        tracksters[i].setRegressedEnergy(0.f);
        tracksters[i].zeroProbabilities();
        tracksterIndices.push_back(i);
        break;
      }
    }
  }

  // do nothing when no trackster passes the selection (3)
  int batchSize = static_cast<int>(tracksterIndices.size());
  if (batchSize == 0) {
    return;
  }

  // create input and output tensors (4)
  tensorflow::TensorShape shape({batchSize, eidNLayers_, eidNClusters_, eidNFeatures_});
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  tensorflow::NamedTensorList inputList = {{eidInputName_, input}};

  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> outputNames;
  if (!eidOutputNameEnergy_.empty()) {
    outputNames.push_back(eidOutputNameEnergy_);
  }
  if (!eidOutputNameId_.empty()) {
    outputNames.push_back(eidOutputNameId_);
  }

  // fill input tensor (5)
  for (int i = 0; i < batchSize; i++) {
    const Trackster &trackster = tracksters[tracksterIndices[i]];

    // per layer, we only consider the first eidNClusters_ clusters in terms of energy, so in order
    // to avoid creating large / nested structures to do the sorting for an unknown number of total
    // clusters, create a sorted list of layer cluster indices to keep track of the filled clusters
    std::vector<int> clusterIndices(trackster.vertices().size());
    for (int k = 0; k < (int)trackster.vertices().size(); k++) {
      clusterIndices[k] = k;
    }
    sort(clusterIndices.begin(), clusterIndices.end(), [&layerClusters, &trackster](const int &a, const int &b) {
      return layerClusters[trackster.vertices(a)].energy() > layerClusters[trackster.vertices(b)].energy();
    });

    // keep track of the number of seen clusters per layer
    std::vector<int> seenClusters(eidNLayers_);

    // loop through clusters by descending energy
    for (const int &k : clusterIndices) {
      // get features per layer and cluster and store the values directly in the input tensor
      const reco::CaloCluster &cluster = layerClusters[trackster.vertices(k)];
      int j = rhtools_.getLayerWithOffset(cluster.hitsAndFractions()[0].first) - 1;
      if (j < eidNLayers_ && seenClusters[j] < eidNClusters_) {
        // get the pointer to the first feature value for the current batch, layer and cluster
        float *features = &input.tensor<float, 4>()(i, j, seenClusters[j], 0);

        // fill features
        *(features++) = float(cluster.energy() / float(trackster.vertex_multiplicity(k)));
        *(features++) = float(std::abs(cluster.eta()));
        *(features) = float(cluster.phi());

        // increment seen clusters
        seenClusters[j]++;
      }
    }

    // zero-fill features of empty clusters in each layer (6)
    for (int j = 0; j < eidNLayers_; j++) {
      for (int k = seenClusters[j]; k < eidNClusters_; k++) {
        float *features = &input.tensor<float, 4>()(i, j, k, 0);
        for (int l = 0; l < eidNFeatures_; l++) {
          *(features++) = 0.f;
        }
      }
    }
  }

  // run the inference (7)
  tensorflow::run(eidSession, inputList, outputNames, &outputs);

  // store regressed energy per trackster (8)
  if (!eidOutputNameEnergy_.empty()) {
    // get the pointer to the energy tensor, dimension is batch x 1
    float *energy = outputs[0].flat<float>().data();

    for (const int &i : tracksterIndices) {
      tracksters[i].setRegressedEnergy(*(energy++));
    }
  }

  // store id probabilities per trackster (8)
  if (!eidOutputNameId_.empty()) {
    // get the pointer to the id probability tensor, dimension is batch x id_probabilities.size()
    int probsIdx = eidOutputNameEnergy_.empty() ? 0 : 1;
    float *probs = outputs[probsIdx].flat<float>().data();

    for (const int &i : tracksterIndices) {
      tracksters[i].setProbabilities(probs);
      probs += tracksters[i].id_probabilities().size();
    }
  }
}

template <typename TILES>
void PatternRecognitionbyFastJet<TILES>::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
  iDesc.add<double>("antikt_radius", 0.09)->setComment("Radius to be used while running the Anti-kt clustering");
  iDesc.add<int>("minNumLayerCluster", 5)->setComment("Not Inclusive");
  iDesc.add<std::string>("eid_input_name", "input");
  iDesc.add<std::string>("eid_output_name_energy", "output/regressed_energy");
  iDesc.add<std::string>("eid_output_name_id", "output/id_probabilities");
  iDesc.add<double>("eid_min_cluster_energy", 1.);
  iDesc.add<int>("eid_n_layers", 50);
  iDesc.add<int>("eid_n_clusters", 10);
}

template class ticl::PatternRecognitionbyFastJet<TICLLayerTiles>;
