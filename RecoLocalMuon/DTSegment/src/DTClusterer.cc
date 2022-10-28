/******* \class DTClusterer *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTClusterer.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
using namespace edm;

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecClusterCollection.h"
#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/* Constructor */
DTClusterer::DTClusterer(const edm::ParameterSet& pset) {
  // Set verbose output
  debug = pset.getUntrackedParameter<bool>("debug");

  // the name of the 1D rec hits collection
  recHits1DToken_ = consumes<DTRecHitCollection>(pset.getParameter<InputTag>("recHits1DLabel"));
  // min number of hits to build a cluster
  theMinHits = pset.getParameter<unsigned int>("minHits");
  // min number of hits to build a cluster
  theMinLayers = pset.getParameter<unsigned int>("minLayers");

  dtGeomToken_ = esConsumes();
  if (debug)
    cout << "[DTClusterer] Constructor called" << endl;

  produces<DTRecClusterCollection>();
}

/* Destructor */
DTClusterer::~DTClusterer() {}

/* Operations */
void DTClusterer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  if (debug)
    cout << "[DTClusterer] produce called" << endl;
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom = setup.getHandle(dtGeomToken_);

  // Get the 1D rechits from the event
  Handle<DTRecHitCollection> allHits;
  event.getByToken(recHits1DToken_, allHits);

  // Create the pointer to the collection which will store the rechits
  auto clusters = std::make_unique<DTRecClusterCollection>();

  // Iterate through all hit collections ordered by LayerId
  DTRecHitCollection::id_iterator dtLayerIt;
  DTSuperLayerId oldSlId;
  for (dtLayerIt = allHits->id_begin(); dtLayerIt != allHits->id_end(); ++dtLayerIt) {
    // The layerId
    DTLayerId layerId = (*dtLayerIt);
    const DTSuperLayerId SLId = layerId.superlayerId();
    if (SLId == oldSlId)
      continue;  // I'm on the same SL as before
    oldSlId = SLId;

    if (debug)
      cout << "Reconstructing the clusters in " << SLId << endl;

    const DTSuperLayer* sl = dtGeom->superLayer(SLId);

    // Get all the rec hit in the same superLayer in which layerId relies
    DTRecHitCollection::range range = allHits->get(DTRangeMapAccessor::layersBySuperLayer(SLId));

    // Fill the vector with the 1D RecHit
    vector<DTRecHit1DPair> pairs(range.first, range.second);
    if (debug)
      cout << "Number of 1D-RecHit pairs " << pairs.size() << endl;
    vector<DTSLRecCluster> clus = buildClusters(sl, pairs);
    if (debug)
      cout << "Number of clusters build " << clus.size() << endl;
    if (!clus.empty())
      clusters->put(sl->id(), clus.begin(), clus.end());
  }
  event.put(std::move(clusters));
}

vector<DTSLRecCluster> DTClusterer::buildClusters(const DTSuperLayer* sl, vector<DTRecHit1DPair>& pairs) const {
  // create a vector of hits with wire position in SL frame
  vector<pair<float, DTRecHit1DPair> > hits = initHits(sl, pairs);

  vector<DTSLRecCluster> result;
  // loop over pairs
  vector<DTRecHit1DPair> adiacentPairs;
  float lastPos = hits.front().first;
  const float cellWidth = 4.2;  // cm
  float sum = 0.;
  float sum2 = 0.;

  for (vector<pair<float, DTRecHit1DPair> >::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    if (debug)
      cout << "Hit: " << (*hit).first << " lastPos: " << lastPos << endl;
    // start from first hits
    // two cells are adiacente if their position is closer than cell width
    if (abs((*hit).first - lastPos) > cellWidth) {
      if (adiacentPairs.size() >= theMinHits && differentLayers(adiacentPairs) >= theMinLayers) {
        // if not, build the cluster with so far collection hits and go on
        float mean = sum / adiacentPairs.size();
        float err2 = sum2 / adiacentPairs.size() - mean * mean;
        DTSLRecCluster cluster(sl->id(), LocalPoint(mean, 0., 0.), LocalError(err2, 0., 0.), adiacentPairs);
        if (debug)
          cout << "Cluster " << cluster << endl;
        result.push_back(cluster);
      }
      // clean the vector
      adiacentPairs.clear();
      sum = 0.;
      sum2 = 0.;
    }
    // if adiacente, add them to a vector
    adiacentPairs.push_back((*hit).second);
    if (debug)
      cout << "adiacentPairs " << adiacentPairs.size() << endl;
    sum += (*hit).first;
    sum2 += (*hit).first * (*hit).first;

    lastPos = (*hit).first;
  }
  // build the last cluster
  if (adiacentPairs.size() >= theMinHits && differentLayers(adiacentPairs) >= theMinLayers) {
    float mean = sum / adiacentPairs.size();
    float err2 = sum2 / adiacentPairs.size() - mean * mean;
    DTSLRecCluster cluster(sl->id(), LocalPoint(mean, 0., 0.), LocalError(err2, 0., 0.), adiacentPairs);
    if (debug)
      cout << "Cluster " << cluster << endl;
    result.push_back(cluster);
  }

  return result;
}

vector<pair<float, DTRecHit1DPair> > DTClusterer::initHits(const DTSuperLayer* sl,
                                                           vector<DTRecHit1DPair>& pairs) const {
  vector<pair<float, DTRecHit1DPair> > result;
  for (vector<DTRecHit1DPair>::const_iterator pair = pairs.begin(); pair != pairs.end(); ++pair) {
    // get wire
    DTWireId wid = (*pair).wireId();
    // get Layer
    const DTLayer* lay = sl->layer(wid.layer());
    // get wire position in SL (only x)
    LocalPoint posInLayer(lay->specificTopology().wirePosition(wid.wire()), 0., 0.);
    LocalPoint posInSL = sl->toLocal(lay->toGlobal(posInLayer));
    // put the pair into result
    result.push_back(make_pair(posInSL.x(), *pair));
  }
  // sorted by x
  sort(result.begin(), result.end(), sortClusterByX());

  return result;
}

unsigned int DTClusterer::differentLayers(vector<DTRecHit1DPair>& hits) const {
  // Count the number of different layers
  int layers = 0;
  unsigned int result = 0;
  for (vector<DTRecHit1DPair>::const_iterator hit = hits.begin(); hit != hits.end(); ++hit) {
    int pos = (1 << ((*hit).wireId().layer() - 1));
    if (!(pos & layers)) {
      result++;
      layers |= pos;
    }
  }
  return result;
}
