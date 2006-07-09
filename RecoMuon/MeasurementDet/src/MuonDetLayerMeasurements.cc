/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  $Date: 2006/07/06 08:18:28 $
 *  $Revision: 1.12 $
 *  \author C. Liu - Purdue University
 *
 */

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

MuonDetLayerMeasurements::MuonDetLayerMeasurements(bool enableDT, bool enableCSC,
						   std::string dtlabel, std::string csclabel): enableDTMeasurement(enableDT),
											       enableCSCMeasurement(enableCSC),
											       theDTRecHitLabel(dtlabel),
											       theCSCRecHitLabel(csclabel)
{
  theEventFlag = false;
  
}

MuonDetLayerMeasurements::~MuonDetLayerMeasurements() {

}

RecHitContainer MuonDetLayerMeasurements::recHits(const GeomDet* geomDet, const edm::Event& iEvent) const {

  RecHitContainer muonRecHits;
  
  DetId geoId = geomDet->geographicalId();
  
  if (geoId.subdetId()  == MuonSubdetId::DT ) {
    if(!enableDTMeasurement) return muonRecHits;

    // Get the DT-Segment collection from the Event
    edm::Handle<DTRecSegment4DCollection> dtRecHits;
    iEvent.getByLabel(theDTRecHitLabel, dtRecHits);  
    
    // Create the ChamberId
    DTChamberId chamberId(geoId.rawId());
    LogDebug("Muon|RecoMuon|MuonDetLayerMeasurements") << "(DT): "<<chamberId<<std::endl;
    
    // Get the DT-Segment which relies on this chamber
    DTRecSegment4DCollection::range  range = dtRecHits->get(chamberId);
    
    // Create the MuonTransientTrackingRechit
    for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit){
      
      MuonTransientTrackingRecHit* muonRecHit = new MuonTransientTrackingRecHit(geomDet, (&(*rechit)));
      muonRecHits.push_back(muonRecHit);
    }
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::CSC) {
    if(!enableCSCMeasurement) return muonRecHits;

    // Get the CSC-Segment collection from the event
    edm::Handle<CSCSegmentCollection> cscSegments;
    iEvent.getByLabel(theCSCRecHitLabel, cscSegments); 

    // Create the chamber Id
    CSCDetId chamberId(geoId.rawId());
    LogDebug("Muon|RecoMuon|MuonDetLayerMeasurements") << "(CSC): "<<chamberId<<std::endl;

    // Get the CSC-Segment which relies on this chamber
    CSCSegmentCollection::range  range = cscSegments->get(chamberId);
    
    // Create the MuonTransientTrackingRecHit
    for (CSCSegmentCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
      
      MuonTransientTrackingRecHit* muonRecHit = new MuonTransientTrackingRecHit(geomDet, (&(*rechit)));
      muonRecHits.push_back(muonRecHit);
    }    
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::RPC ) {
    RPCDetId chamberId(geoId.rawId());
    LogDebug("Muon|RecoMuon|MuonDetLayerMeasurements") << "(RPC): "<<chamberId<<std::endl;
  }
  
  else {
    // wrong type
    edm::LogWarning("MuonDetLayerMeasurements")<<"The DetLayer is not a valid Muon DetLayer. ";
  }
  return muonRecHits;
}


MeasurementContainer
MuonDetLayerMeasurements::measurements( const DetLayer* layer,
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop,
					const MeasurementEstimator& est,
					const edm::Event& iEvent) const {
  
  MeasurementContainer result;
  
  std::vector<DetWithState> dss = layer->compatibleDets(startingState, prop, est);
  
  for (std::vector<DetWithState>::const_iterator ids = dss.begin(); ids !=dss.end(); ids++){
    
    // Get the Segments which relies on the GeomDet given by compatibleDets
    RecHitContainer muonRecHits = recHits( (*ids).first , iEvent);
    
    // Create the Trajectory Measurement
    for(RecHitContainer::const_iterator rechit = muonRecHits.begin();
	rechit != muonRecHits.end(); ++rechit)
      result.push_back(TrajectoryMeasurement((*ids).second, *rechit, 0, layer)); 
  }
  return result;
}

MeasurementContainer
MuonDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					    const TrajectoryStateOnSurface& theStateOnDet,
					    const TrajectoryStateOnSurface& startingState,
					    const Propagator& prop,
					    const MeasurementEstimator& est,
					    const edm::Event& iEvent) const {
  MeasurementContainer result;
  RecHitContainer rhs = recHits(layer, iEvent);
  for (RecHitContainer::const_iterator irh = rhs.begin(); irh!=rhs.end(); irh++) {
    if (est.estimate(theStateOnDet, (**irh)).first)
      result.push_back(TrajectoryMeasurement(theStateOnDet,(*irh),0,layer));
  }

  return result;
}

///measurements method if already got the Event 
MeasurementContainer
MuonDetLayerMeasurements::measurements( const DetLayer* layer,
					const TrajectoryStateOnSurface& startingState,
					const Propagator& prop,
					const MeasurementEstimator& est) const {
  MeasurementContainer result;
  if (theEventFlag) return measurements(layer, startingState, prop, est, *theEvent);
  else return result;
}

///fastMeasurements method if already got the Event
MeasurementContainer
MuonDetLayerMeasurements::fastMeasurements( const DetLayer* layer,
					    const TrajectoryStateOnSurface& theStateOnDet,
					    const TrajectoryStateOnSurface& startingState,
					    const Propagator& prop,
					    const MeasurementEstimator& est) const {
  MeasurementContainer result;
  if (theEventFlag) return fastMeasurements(layer, theStateOnDet, startingState, prop, est, *theEvent); 
  else return result;
}

///set event
void MuonDetLayerMeasurements::setEvent(const edm::Event& event) {
  theEvent = &event;
  theEventFlag = true;
}

RecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer, const edm::Event& iEvent) const
{
  RecHitContainer rhs;
  
  Module mtype = layer->module();

  if (mtype == dt ) {
    if(!enableDTMeasurement) return rhs;

    edm::Handle<DTRecSegment4DCollection> dtRecHits;
    iEvent.getByLabel(theDTRecHitLabel, dtRecHits);  
    
    std::vector <const GeomDet*> gds = layer->basicComponents();
    for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {
      DTChamberId chamberId((*igd)->geographicalId().rawId());
      DTRecSegment4DCollection::range  range = dtRecHits->get(chamberId);
      for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit){
	MuonTransientTrackingRecHit* gttrh = new MuonTransientTrackingRecHit((*igd), (&(*rechit)));
	rhs.push_back(gttrh);
      }//for DTRecSegment4DCollection
    }// for GeomDet
  }
  else if (mtype == csc ) {
    if(!enableCSCMeasurement) return rhs;
    
    edm::Handle<CSCSegmentCollection> cscSegments;
    iEvent.getByLabel(theCSCRecHitLabel, cscSegments); 

    std::vector <const GeomDet*> gds = layer->basicComponents();

    for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {
      CSCDetId id((*igd)->geographicalId().rawId());
      CSCSegmentCollection::range  range = cscSegments->get(id);
      for (CSCSegmentCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){

	MuonTransientTrackingRecHit* gttrh = new MuonTransientTrackingRecHit((*igd), (&(*rechit)));
	rhs.push_back(gttrh);
      }//for CSCSegmentCollection
    }// for GeomDet
  }else if (mtype == rpc ) {

  }else {
    //wrong type
    edm::LogInfo("MuonDetLayerMeasurements")<<"The DetLayer is not a valid Muon DetLayer. ";
  }
  return rhs;
}

RecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer) const
{
  RecHitContainer result;
  if (theEventFlag) return recHits(layer, *theEvent);
  else return result;
}

