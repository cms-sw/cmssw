// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 05/2024

#include <vector>
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PatternRecognitionbyPassthrough.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackstersPCA.h"

using namespace ticl;

template <typename TILES>
PatternRecognitionbyPassthrough<TILES>::PatternRecognitionbyPassthrough(const edm::ParameterSet &conf,
                                                                        edm::ConsumesCollector iC)
    : PatternRecognitionAlgoBaseT<TILES>(conf, iC), caloGeomToken_(iC.esConsumes<CaloGeometry, CaloGeometryRecord>()) {}

template <typename TILES>
void PatternRecognitionbyPassthrough<TILES>::makeTracksters(
    const typename PatternRecognitionAlgoBaseT<TILES>::Inputs &input,
    std::vector<Trackster> &result,
    std::unordered_map<int, std::vector<int>> &seedToTracksterAssociation) {
  // Get the geometry setup
  edm::EventSetup const &es = input.es;
  const CaloGeometry &geom = es.getData(caloGeomToken_);
  rhtools_.setGeometry(geom);

  // Clear the result vector
  result.clear();

  // Iterate over all layer clusters
  for (size_t i = 0; i < input.layerClusters.size(); ++i) {
    if (input.mask[i] == 0.) {
      continue;  // Skip masked clusters
    }

    // Create a new trackster for each layer cluster
    Trackster trackster;
    trackster.vertices().push_back(i);
    trackster.vertex_multiplicity().push_back(1);

    // Add the trackster to the result vector
    result.push_back(trackster);
  }

  // Assign PCA to tracksters
  ticl::assignPCAtoTracksters(result,
                              input.layerClusters,
                              input.layerClustersTime,
                              rhtools_.getPositionLayer(rhtools_.lastLayerEE(false), false).z(),
                              rhtools_,
                              false);

  // Log the number of tracksters created
  if (PatternRecognitionAlgoBaseT<TILES>::algo_verbosity_ > VerbosityLevel::Advanced) {
    edm::LogVerbatim("PatternRecognitionbyPassthrough") << "Created " << result.size() << " tracksters";
  }
}

template <typename TILES>
void PatternRecognitionbyPassthrough<TILES>::fillPSetDescription(edm::ParameterSetDescription &iDesc) {
  iDesc.add<int>("algo_verbosity", 0);
}

// Explicitly instantiate the templates
template class ticl::PatternRecognitionbyPassthrough<TICLLayerTiles>;
template class ticl::PatternRecognitionbyPassthrough<TICLLayerTilesHFNose>;
