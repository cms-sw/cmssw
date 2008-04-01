
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "CLHEP/Vector/RotationInterfaces.h" 

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"


TrackerValidationVariables::TrackerValidationVariables() {}


TrackerValidationVariables::TrackerValidationVariables(const edm::EventSetup& es, const edm::ParameterSet& iSetup) 
  : conf_(iSetup), fBfield(4.06)
{
  es.get<TrackerDigiGeometryRecord>().get( tkgeom );
  //es.get<SiStripDetCablingRcd>().get( SiStripDetCabling_ );
}

TrackerValidationVariables::~TrackerValidationVariables() {}

void 
TrackerValidationVariables::fillHitQuantities(const edm::Event& iEvent, 
				      std::vector<AVHitStruct>& v_avhitout )
{
  edm::Handle<std::vector<Trajectory> > trajCollectionHandle;
  iEvent.getByLabel(conf_.getParameter<std::string>("trajectoryInput"),trajCollectionHandle);
  
  TrajectoryStateCombiner tsoscomb;
  edm::LogVerbatim("TrackerValidationVariables") << "trajColl->size(): " << trajCollectionHandle->size() ;
  for(std::vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(), itEnd = trajCollectionHandle->end(); 
      it!=itEnd;++it){
    std::vector<TrajectoryMeasurement> tmColl = it->measurements();
    for(std::vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(), itTrajEnd = tmColl.end(); 
	itTraj != itTrajEnd; ++itTraj) {


      if(! itTraj->updatedState().isValid()) continue;
      
      TrajectoryStateOnSurface theCombinedPredictedState = tsoscomb.combine(itTraj->backwardPredictedState(), 
									    itTraj->forwardPredictedState() );      
      TransientTrackingRecHit::ConstRecHitPointer hit = itTraj->recHit();
      if(! hit->isValid()) {
	continue; 
      } else {
	AVHitStruct hitStruct;
	const DetId & hit_detId = hit->geographicalId();
	uint IntRawDetID = (hit_detId.rawId());	
	//
	TrajectoryStateOnSurface tsos = tsoscomb( itTraj->forwardPredictedState(), itTraj->backwardPredictedState() );
	
	align::LocalVector res = tsos.localPosition() - hit->localPosition();
	LocalError err1 = tsos.localError().positionError();
	LocalError err2 = hit->localPositionError();
	
	float errX = std::sqrt( err1.xx() + err2.xx() );
	float errY = std::sqrt( err1.yy() + err2.yy() );
	
	LogDebug("TrackerValidationVariables") << "@SUB=TrackerValidationVariables::fillHitQuantities" 
					       << "Residual x/y " << res.x() << '/' << res.y() 
					       << ", Error x/y " << errX << '/' << errY;
		
	hitStruct.resX = res.x();
	hitStruct.resY = res.y();
	hitStruct.resErrX = errX;
	hitStruct.resErrY = errY;
	hitStruct.rawDetId = IntRawDetID;
	hitStruct.phi = tsos.globalDirection().phi();

	
	v_avhitout.push_back(hitStruct);
      } 
    } 
  }  

} 

void 
TrackerValidationVariables::fillTrackQuantities(const edm::Event& iEvent,
					std::vector<AVTrackStruct>& v_avtrackout)
{
  // get track collection from the event
  edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("Tracks");
  edm::Handle<reco::TrackCollection> RecoTracks;
  iEvent.getByLabel(TkTag,RecoTracks);
  edm::LogInfo("TrackInfoAnalyzerExample")<<"track collection size "<< RecoTracks->size();
  
  // Put here all track based quantities such as eta, phi, pt,.... 
  int i=0;
  for( reco::TrackCollection::const_iterator RecoTrack = RecoTracks->begin(), RecoTrackEnd = RecoTracks->end();
       RecoTrack !=RecoTrackEnd ; ++i, ++RecoTrack) {
    AVTrackStruct trackStruct;
    trackStruct.pt = RecoTrack->pt();
    trackStruct.px = RecoTrack->px();
    trackStruct.py = RecoTrack->py();
    trackStruct.pz = RecoTrack->pz();
    trackStruct.eta = RecoTrack->eta();
    trackStruct.phi = RecoTrack->phi();
    trackStruct.chi2 = RecoTrack->chi2();
    trackStruct.normchi2 = RecoTrack->normalizedChi2();
    trackStruct.kappa = -RecoTrack->charge()*0.002998*fBfield/RecoTrack->pt();
    trackStruct.charge = RecoTrack->charge();
    trackStruct.d0 = RecoTrack->d0();
    trackStruct.dz = RecoTrack->dz();
    v_avtrackout.push_back(trackStruct);
  }

}

