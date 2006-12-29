/** \class MuonDetLayerMeasurements
 *  The class to access recHits and TrajectoryMeasurements from DetLayer.
 *
 *  $Date: 2006/08/03 13:39:10 $
 *  $Revision: 1.19 $
 *  \author C. Liu, R. Bellan, N. Amapane
 *
 */

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


//typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
//typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;
typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;


MuonDetLayerMeasurements::MuonDetLayerMeasurements(bool enableDT, bool enableCSC, bool enableRPC,
						   std::string dtlabel, std::string csclabel,
						   std::string rpclabel): enableDTMeasurement(enableDT),
									  enableCSCMeasurement(enableCSC),
									  enableRPCMeasurement(enableRPC),
									  theDTRecHitLabel(dtlabel),
									  theCSCRecHitLabel(csclabel),
									  theRPCRecHitLabel(rpclabel)
{
  theEventFlag = false;
  
}

MuonDetLayerMeasurements::~MuonDetLayerMeasurements() {

}

MuonTransientTrackingRecHit::MuonRecHitContainer MuonDetLayerMeasurements::recHits(const GeomDet* geomDet, const edm::Event& iEvent) const {

  MuonRecHitContainer muonRecHits;
  
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
    DTRecSegment4DCollection::range range = dtRecHits->get(chamberId);
    
    // Create the MuonTransientTrackingRechit
    for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit){
      
      MuonRecHitPointer muonRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit)));
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
    CSCSegmentCollection::range range = cscSegments->get(chamberId);
    
    // Create the MuonTransientTrackingRecHit
    for (CSCSegmentCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
      
      MuonRecHitPointer muonRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit)));
      muonRecHits.push_back(muonRecHit);
    }    
  }
  
  else if (geoId.subdetId()  == MuonSubdetId::RPC ) {
    if(!enableRPCMeasurement) return muonRecHits;
    
    // Get the CSC-Segment collection from the event
    edm::Handle<RPCRecHitCollection> rpcRecHits;
    iEvent.getByLabel(theRPCRecHitLabel, rpcRecHits); 
    
    // Create the chamber Id
    RPCDetId chamberId(geoId.rawId());
    LogDebug("Muon|RecoMuon|MuonDetLayerMeasurements") << "(RPC): "<<chamberId<<std::endl;
    
    // Get the RPC-Segment which relies on this chamber
    RPCRecHitCollection::range range = rpcRecHits->get(chamberId);
    
    // Create the MuonTransientTrackingRecHit
    for (RPCRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
      
      MuonRecHitPointer muonRecHit = MuonTransientTrackingRecHit::specificBuild(geomDet, (&(*rechit)));
      muonRecHits.push_back(muonRecHit);
    }
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
  LogDebug("RecoMuon")<<"compatibleDets: "<<dss.size()<<std::endl;
  
  for (std::vector<DetWithState>::const_iterator ids = dss.begin(); ids !=dss.end(); ids++){
    
    // Get the Segments which relies on the GeomDet given by compatibleDets
    MuonRecHitContainer muonRecHits = recHits( (*ids).first , iEvent);
    
    // Create the Trajectory Measurement
    for(MuonRecHitContainer::iterator rechit = muonRecHits.begin();
	rechit != muonRecHits.end(); ++rechit) {
      result.push_back(TrajectoryMeasurement((*ids).second, (*rechit).get(), 0, layer)); 
    }
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
  MuonRecHitContainer rhs = recHits(layer, iEvent);
  for (MuonRecHitContainer::const_iterator irh = rhs.begin(); irh!=rhs.end(); irh++) {
    if (est.estimate(theStateOnDet, (**irh)).first)
      result.push_back(TrajectoryMeasurement(theStateOnDet,(*irh).get(),0,layer));
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

MuonRecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer, const edm::Event& iEvent) const
{
  MuonRecHitContainer rhs;
  
  GeomDetEnumerators::SubDetector mtype = layer->subDetector();

  if (mtype == GeomDetEnumerators::DT ) {
    if(!enableDTMeasurement) return rhs;

    // Get the DT-Segment collection from the event
    edm::Handle<DTRecSegment4DCollection> dtRecHits;
    iEvent.getByLabel(theDTRecHitLabel, dtRecHits);  
    
    std::vector <const GeomDet*> gds = layer->basicComponents();
    
    for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {

      // Create the chamber Id
      DTChamberId chamberId((*igd)->geographicalId().rawId());

      // Get the DT-Segment which relies on this chamber
      DTRecSegment4DCollection::range  range = dtRecHits->get(chamberId);

       // Create the MuonTransientTrackingRecHit
      for (DTRecSegment4DCollection::const_iterator rechit = range.first; rechit!=range.second;++rechit){
	MuonRecHitPointer gttrh = MuonTransientTrackingRecHit::specificBuild((*igd), (&(*rechit)));
	rhs.push_back(gttrh);
      }
    }
  }
  else if (mtype == GeomDetEnumerators::CSC ) {
    if(!enableCSCMeasurement) return rhs;
    
    // Get the CSC-Segment collection from the event
    edm::Handle<CSCSegmentCollection> cscSegments;
    iEvent.getByLabel(theCSCRecHitLabel, cscSegments); 

    std::vector <const GeomDet*> gds = layer->basicComponents();
    
    for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {

      // Create the chamber Id
      CSCDetId id((*igd)->geographicalId().rawId());

      // Get the RPC-Segment which relies on this chamber
      CSCSegmentCollection::range  range = cscSegments->get(id);

      // Create the MuonTransientTrackingRecHit
      for (CSCSegmentCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
	MuonRecHitPointer gttrh = MuonTransientTrackingRecHit::specificBuild((*igd), (&(*rechit)));
	rhs.push_back(gttrh);
      }
    }
  }
  else if ( (mtype == GeomDetEnumerators::RPCBarrel) || (mtype == GeomDetEnumerators::RPCEndcap) ) {
    if(!enableRPCMeasurement) return rhs;
    
    edm::Handle<RPCRecHitCollection> rpcRecHits;
    iEvent.getByLabel(theRPCRecHitLabel, rpcRecHits); 
    
    std::vector <const GeomDet*> gds = layer->basicComponents();

    for (std::vector<const GeomDet*>::const_iterator igd = gds.begin(); igd != gds.end(); igd++) {

      // Create the chamber Id
      RPCDetId id((*igd)->geographicalId().rawId());
      
      // Get the RPC-Segment which relies on this chamber
      RPCRecHitCollection::range  range = rpcRecHits->get(id);
      
      // Create the MuonTransientTrackingRecHit
      for (RPCRecHitCollection::const_iterator rechit = range.first; rechit!=range.second; ++rechit){
	MuonRecHitPointer gttrh = MuonTransientTrackingRecHit::specificBuild((*igd), (&(*rechit)));
	rhs.push_back(gttrh);
      }
    }
  }
  else {
    //wrong type
    edm::LogInfo("MuonDetLayerMeasurements")<<"The DetLayer is not a valid Muon DetLayer. ";
  }
  return rhs;
}

MuonRecHitContainer MuonDetLayerMeasurements::recHits(const DetLayer* layer) const
{
  MuonRecHitContainer result;
  if (theEventFlag) return recHits(layer, *theEvent);
  else return result;
}

