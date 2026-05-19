// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 05/2024

#include <vector>
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyRecovery.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackstersPCA.h"

using namespace ticl;

template <typename TILES>
PatternRecognitionbyRecovery<TILES>::PatternRecognitionbyRecovery(const edm::ParameterSet &conf,
                                                                  edm::ConsumesCollector iC)
    : PatternRecognitionAlgoBaseT<TILES>(conf, iC) {}

template <typename TILES>
void PatternRecognitionbyRecovery<TILES>::setGeometry(hgcal::RecHitTools const &rhtools) {
  this->rhtools_ = &rhtools;
  z_limit_em_ = std::abs(this->rhtools_->getPositionLayer(this->rhtools_->lastLayerEE(false), false).z());
  this->geometryReady_ = true;
}

template <typename TILES>
void PatternRecognitionbyRecovery<TILES>::makeTracksters(
    const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
    std::vector<Trackster> &result,
    std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  if (UNLIKELY(!this->geometryReady_ || this->rhtools_ == nullptr)) {
    throw cms::Exception("LogicError")
        << "PatternRecognitionbyRecovery::setGeometry() must be called in beginRun() before makeTracksters().";
  }

  // Clear the result vector
  result.clear();

  result.reserve(input.layerClusters.size() / 16);  // Heuristic

  // Iterate over all layer clusters
  for (unsigned int i = 0; i < input.layerClusters.size(); ++i) {
    if (input.mask[i] == 0.f) {
      continue;  // Skip masked clusters
    }
    // Create a new trackster for each layer cluster
    result.emplace_back();
    auto &trackster = result.back();
    auto &v = trackster.vertices();
    v.clear();
    v.reserve(1);
    v.push_back(i);

    auto &mult = trackster.vertex_multiplicity();
    mult.clear();
    mult.reserve(1);
    mult.push_back(1);
    const auto &lc = input.layerClusters[i];
    const auto timePair = input.layerClustersTime.get(i);
    trackster.setTimeAndError(timePair.first, timePair.second);
    trackster.setRawEnergy(lc.energy());
    trackster.setBarycenter({float(lc.x()), float(lc.y()), float(lc.z())});
    trackster.calculateRawPt();
    const float z = lc.z();
    if (z <= z_limit_em_ && z >= -z_limit_em_) {
      trackster.setRawEmEnergy(lc.energy());
      trackster.calculateRawEmPt();
    }
  }
  result.shrink_to_fit();
  // Log the number of tracksters created
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyRecovery") << "Created " << result.size() << " tracksters";
  }
}

template <typename TILES>
void PatternRecognitionbyRecovery<TILES>::filter(std::vector<Trackster> &output,
                                                 const std::vector<Trackster> &inTracksters,
                                                 const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
                                                 std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  output = inTracksters;
}

template <typename TILES>
void PatternRecognitionbyRecovery<TILES>::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
}

// Explicitly instantiate the templates
template class ticl::PatternRecognitionbyRecovery<TICLLayerTiles>;
template class ticl::PatternRecognitionbyRecovery<TICLLayerTilesHFNose>;
