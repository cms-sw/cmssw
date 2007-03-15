#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducerAlgorithm.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackingRecHitInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Utilities/General/interface/CMSexception.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"


void TrackInfoProducerAlgorithm::run(std::vector<Trajectory>::const_iterator  traj_iterator,reco::TrackRef track,
				     reco::TrackInfo &outputFwd,
				     reco::TrackInfo &outputBwd,
				     reco::TrackInfo &outputUpdated, 
				     reco::TrackInfo &outputCombined,        const TrackerGeometry * tracker)
{
  //  edm::LogInfo("TrackInfoProducer") << "Number of Trajectories: "<<inputcoll->size();

  //  std::vector<Trajectory>::const_iterator traj_iterator;
  
  // for(traj_iterator=inputcoll->begin();traj_iterator!=inputcoll->end();traj_iterator++){//loop on trajectories
    std::vector<TrajectoryMeasurement> measurements =traj_iterator->measurements();
    
    std::vector<TrajectoryMeasurement>::iterator traj_mes_iterator;
    //edm::LogInfo("TrackInfoProducer") << "Number of Measurements: "<<measurements.size();
    reco::TrackInfo::TrajectoryInfo fwdtrajinfo;
    reco::TrackInfo::TrajectoryInfo bwdtrajinfo;
    reco::TrackInfo::TrajectoryInfo updatedtrajinfo;
    reco::TrackInfo::TrajectoryInfo combinedtrajinfo;
    int nhit=0;
    for(traj_mes_iterator=measurements.begin();traj_mes_iterator!=measurements.end();traj_mes_iterator++){//loop on measurements
      TrajectoryStateOnSurface  fwdtsos=traj_mes_iterator->forwardPredictedState();
      TrajectoryStateOnSurface  bwdtsos=traj_mes_iterator->backwardPredictedState();
      TrajectoryStateOnSurface  updatedtsos=traj_mes_iterator->updatedState();
      TrajectoryStateCombiner statecombiner;
      TrajectoryStateOnSurface  combinedtsos=statecombiner.combine(fwdtsos, bwdtsos);

      ConstRecHitPointer ttrh=traj_mes_iterator->recHit();
      LocalPoint pos;
      if (ttrh->isValid())pos=ttrh->hit()->localPosition() ;
      nhit++;
      unsigned int detid=ttrh->hit()->geographicalId().rawId();
      
      trackingRecHit_iterator thehit;
      TrackingRecHitRef thehitref;
      int i=0,j=0;
      //edm::LogInfo("TrackInfoProducer") <<"Rechit size: "<<rechits->product()->size();
      for (thehit=track->recHitsBegin();thehit!=track->recHitsEnd();thehit++){
	i++;
	LocalPoint hitpos;
	if ((*thehit)->isValid())hitpos=(*thehit)->localPosition();
	if((*thehit)->geographicalId().rawId()==detid&&
	   (hitpos - pos).mag() < 1e-4)
	  {
	    thehitref=(*thehit);
	    //	  edm::LogInfo("TrackInfoProducer") << "Found a rechit ";
	    j++;
	    break;
	  }
      }
      TrajectoryStateTransform tsostransform;
      PTrajectoryStateOnDet* fwdptsod=tsostransform.persistentState( fwdtsos,detid);
      PTrajectoryStateOnDet* bwdptsod=tsostransform.persistentState( bwdtsos,detid);
      PTrajectoryStateOnDet* updatedptsod=tsostransform.persistentState( updatedtsos,detid);
      PTrajectoryStateOnDet* combinedptsod=tsostransform.persistentState( combinedtsos,detid);
      

      const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>( &*(thehitref));
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>( &*(thehitref));
      //const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>( &*(thehitref));
      reco::TrackingRecHitInfo::RecHitType type=reco::TrackingRecHitInfo::Single;
      LocalVector monofwd, stereofwd;
      LocalVector monobwd, stereobwd;
      LocalVector monoco, stereoco;
      LocalVector monoup, stereoup;
      if(matchedhit){
	type=reco::TrackingRecHitInfo::Matched;
	GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
	
	GlobalVector gtrkdirfwd=gdet->toGlobal(fwdptsod->parameters().momentum());
	GlobalVector gtrkdirbwd=gdet->toGlobal(bwdptsod->parameters().momentum());
	GlobalVector gtrkdirup=gdet->toGlobal(updatedptsod->parameters().momentum());
	GlobalVector gtrkdirco=gdet->toGlobal(combinedptsod->parameters().momentum());
	
	const GeomDetUnit * monodet=gdet->monoDet();
	
	monofwd=monodet->toLocal(gtrkdirfwd);
	monobwd=monodet->toLocal(gtrkdirbwd);
	monoup=monodet->toLocal(gtrkdirup);
	monoco=monodet->toLocal(gtrkdirco);

	const GeomDetUnit * stereodet=gdet->stereoDet();
	
	stereofwd=stereodet->toLocal(gtrkdirfwd);
	stereobwd=stereodet->toLocal(gtrkdirbwd);
	stereoup=stereodet->toLocal(gtrkdirup);
	stereoco=stereodet->toLocal(gtrkdirco);
      }
      else if(phit){
	type=reco::TrackingRecHitInfo::Projected;
	GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(phit->geographicalId());
	
	GlobalVector gtrkdirfwd=gdet->toGlobal(fwdptsod->parameters().momentum());
	GlobalVector gtrkdirbwd=gdet->toGlobal(bwdptsod->parameters().momentum());
	GlobalVector gtrkdirup=gdet->toGlobal(updatedptsod->parameters().momentum());
	GlobalVector gtrkdirco=gdet->toGlobal(combinedptsod->parameters().momentum());
	const SiStripRecHit2D&  originalhit=phit->originalHit();
	const GeomDetUnit * det;
	if(!StripSubdetector(originalhit.geographicalId().rawId()).stereo()){
	  det=gdet->monoDet();
	  monofwd= det->toLocal(gtrkdirfwd);
	  monobwd= det->toLocal(gtrkdirbwd);
	  monoup=  det->toLocal(gtrkdirup);
	  monoco=  det->toLocal(gtrkdirco);
	}
	else{
	  det=gdet->stereoDet();
	  stereofwd= det->toLocal(gtrkdirfwd);
	  stereobwd= det->toLocal(gtrkdirbwd);
	  stereoup=  det->toLocal(gtrkdirup);
	  stereoco=  det->toLocal(gtrkdirco);
	}
      }
      
      reco::TrackingRecHitInfo tkRecHitInfofwd =reco::TrackingRecHitInfo(type, std::make_pair(monofwd,stereofwd),*fwdptsod );
      reco::TrackingRecHitInfo tkRecHitInfobwd =reco::TrackingRecHitInfo(type, std::make_pair(monobwd,stereobwd),*bwdptsod );
      reco::TrackingRecHitInfo tkRecHitInfoup =reco::TrackingRecHitInfo(type, std::make_pair(monoup,stereoup),*updatedptsod );
      reco::TrackingRecHitInfo tkRecHitInfoco =reco::TrackingRecHitInfo(type, std::make_pair(monoco,stereoco),*combinedptsod );
      
      
      
      if(j!=0){
	fwdtrajinfo.insert(make_pair(thehitref,tkRecHitInfofwd));
	bwdtrajinfo.insert(make_pair(thehitref,tkRecHitInfobwd));
	updatedtrajinfo.insert(make_pair(thehitref,tkRecHitInfoup));
	combinedtrajinfo.insert(make_pair(thehitref,tkRecHitInfoco));
      }
      //      else  edm::LogInfo("TrackInfoProducer") << "RecHit not associated ";
    }
    //edm::LogInfo("TrackInfoProducer") << "Found "<<nhit<< " hits";
    //if(fwdtrajinfo.size()!=nhit) edm::LogInfo("TrackInfoProducer") << "Number of trackinfos  "<<fwdtrajinfo.size()<< " doesn't match!";
    outputFwd=reco::TrackInfo((traj_iterator->seed()),fwdtrajinfo);
    outputBwd=reco::TrackInfo((traj_iterator->seed()),bwdtrajinfo);
    outputUpdated=reco::TrackInfo((traj_iterator->seed()),updatedtrajinfo);
    outputCombined=reco::TrackInfo((traj_iterator->seed()),combinedtrajinfo);
}
