/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  $Date: $
 *  $Revision: $
 *  \author C. Liu - Purdue University
 *
 */

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 

MuonDetLayerMeasurements::MuonDetLayerMeasurements() {

}

MuonDetLayerMeasurements::~MuonDetLayerMeasurements() {

}

RecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer, const edm::Event& iEvent) const
{
  RecHitContainer rhs;
  
  Module mtype = layer->module();
  if (mtype == dt ) {
     edm::Handle<DTRecSegment4DCollection> dtRecHits;
     iEvent.getByLabel("recseg4dbuilder", dtRecHits);  //FIXME

     DTRecSegment4DCollection::id_iterator detUnitIt;
     for (detUnitIt = dtRecHits->id_begin();detUnitIt != dtRecHits->id_end();
          ++detUnitIt){
                 DTRecSegment4DCollection::range  range = dtRecHits->get((*detUnitIt));
                 for (DTRecSegment4DCollection::const_iterator rechit = range.first;
                      rechit!=range.second;++rechit){
                          DetId id2 = rechit->geographicalId();
                          bool idmatch= false;
                          int i=0; 
                          std::vector <const GeomDet*> gds = layer->basicComponents();
                          for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {
                              DetId id = (*igd)->geographicalId();
                              i++; 
                              if (id==id2) {
                                    idmatch = true;
                                    break;
                              }
                           }
                          if (!idmatch) {
                              continue;
                            }  
                          const GeomDet * oneGD = *(gds.begin()+i-1);   
                          if (oneGD == 0) { 
                              continue;
                           }
                          TransientTrackingRecHit* gttrh = new GenericTransientTrackingRecHit(oneGD, (&(*rechit)));
                          rhs.push_back(gttrh);
                }//for DTSegment4D
       }// for detUnit
  }else if (mtype == csc ) {
     edm::Handle<CSCSegmentCollection> cscSegments;
     iEvent.getByLabel("segmentbuilder", cscSegments); 
     for (CSCSegmentCollection::const_iterator cscSeg = cscSegments->begin();cscSeg != cscSegments->end();
          ++cscSeg){
                          DetId id2 = cscSeg->geographicalId();
                          bool idmatch= false;
                          int i=0;
                          std::vector <const GeomDet*> gds = layer->basicComponents();
                          for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {
                              DetId id = (*igd)->geographicalId();
                              i++;
                              if (id==id2) {
                                    idmatch = true;
                                    break;
                              }
                           }
                          if (!idmatch) break;
                          const GeomDet * oneGD = *(gds.begin()+i-1);
                          std::vector<const TrackingRecHit*> trhs = cscSeg->recHits(); 
                          for (std::vector<const TrackingRecHit*>::const_iterator itt= trhs.begin(); itt !=trhs.end(); itt++ ) {
                          if (!(*itt)->isValid()) continue;
                          TransientTrackingRecHit* gttrh = new GenericTransientTrackingRecHit(oneGD, (*itt));
                          rhs.push_back(gttrh);
                          }
                }//for CSCSegment
  }else if (mtype == rpc ) {

  }else {
      //wrong type
  }
  return rhs;
}

   MeasurementContainer
   MuonDetLayerMeasurements::measurements( const DetLayer& layer,
                 const TrajectoryStateOnSurface& startingState,
                 const Propagator& prop,
                 const MeasurementEstimator& est,
                 const edm::Event& iEvent) const {
      MeasurementContainer result;
      return result;
   }

  MeasurementContainer
  MuonDetLayerMeasurements::fastMeasurements( const DetLayer& layer,
                    const TrajectoryStateOnSurface& theStateOnDet,
                    const TrajectoryStateOnSurface& startingState,
                    const Propagator& prop,
                    const MeasurementEstimator& est,
                    const edm::Event& iEvent) const {
      MeasurementContainer result;
      return result;
   }

