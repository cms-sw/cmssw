/******* \class DTMeanTimer *******
 *
 * Description:
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 22/11/2006 13:14:27 CET $
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTMeanTimer.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
using namespace edm;

/* C++ Headers */
using namespace std;

/* ====================================================================== */

/* Constructor */ 
DTMeanTimer::DTMeanTimer(const DTSuperLayer* sl,
                         Handle<DTRecHitCollection>& hits,
                         const EventSetup& eventSetup,
                         DTTTrigBaseSync* sync) {

  // store the digis in container separated per layer, with the time itself and
  // the wire Id: map[int wire]=double time;

  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  for (DTRecHitCollection::const_iterator hit = hits->begin();
       hit!=hits->end();  ++hit) {
    // get only this SL hits.
    if ((*hit).wireId().superlayerId()!=sl->id() ) continue;

    DTWireId wireId = (*hit).wireId();

    float ttrig = sync->offset(wireId);

    float time = (*hit).digiTime() - ttrig;

    hitsLay[wireId.layer()-1][wireId.wire()]=time;
  }

  //get number of wires for the 4 layers
  int nWiresSL=0;
  for (int l=1; l<=4; ++l) {
    int nWires = sl->layer(l)->specificTopology().channels();
    if (nWires > nWiresSL) nWiresSL=nWires;
  }
  theNumWires = nWiresSL;
}

DTMeanTimer::DTMeanTimer(const DTSuperLayer* sl,
                         vector<DTRecHit1D>& hits,
                         const edm::EventSetup& eventSetup,
                         DTTTrigBaseSync* sync) {
  // store the digis in container separated per layer, with the time itself and
  // the wire Id: map[int wire]=double time;

  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  for (vector<DTRecHit1D>::const_iterator hit = hits.begin();
       hit!=hits.end();  ++hit) {
    // get only this SL hits.
    if ((*hit).wireId().superlayerId()!=sl->id() ) continue;

    DTWireId wireId = (*hit).wireId();

    float ttrig = sync->offset(wireId);

    float time = (*hit).digiTime() - ttrig;

    hitsLay[wireId.layer()-1][wireId.wire()]=time;
  }

  //get number of wires for the 4 layers
  int nWiresSL=0;
  for (int l=1; l<=4; ++l) {
    int nWires = sl->layer(l)->specificTopology().channels();
    if (nWires > nWiresSL) nWiresSL=nWires;
  }
  theNumWires = nWiresSL;
}

/* Destructor */ 
DTMeanTimer::~DTMeanTimer() {
}

/* Operations */ 
vector<double> DTMeanTimer::run() const {

  // the MT can be build from layers 1,2,3 or 2,3,4
  vector<double> res123 = computeMT(hitsLay[0],hitsLay[1],hitsLay[2]);
  vector<double> res234 = computeMT(hitsLay[1],hitsLay[2],hitsLay[3]);

  vector<double> result(res123);
  result.insert(result.end(),res234.begin(),res234.end());
  return result;

}

vector<double> DTMeanTimer::computeMT(hitColl hits1,
                                      hitColl hits2,
                                      hitColl hits3) const {
  vector<double> result;
  //loop over wires and get the triplets if they exist
  for (int w=1; w<=theNumWires; ++w) {
    // check if we have digis in cell w for the 3 layers
    if (hits1.find(w)!=hits1.end() &&
        hits2.find(w)!=hits2.end() &&
        hits3.find(w)!=hits3.end() ) {

      result.push_back( tMax( hits1[w], hits2[w], hits3[w]));
    }
  }
  return result;
}

double DTMeanTimer::tMax(const double& t1, const double& t2, const double& t3) const {
  return (t2+ (t1+t3)/2.);
}
