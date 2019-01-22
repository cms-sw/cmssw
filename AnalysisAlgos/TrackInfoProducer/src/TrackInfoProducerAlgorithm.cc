#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducerAlgorithm.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackingRecHitInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

using namespace reco;

void TrackInfoProducerAlgorithm::run(const edm::Ref<std::vector<Trajectory> > traj_iterator,TrackRef track,
				     TrackInfo &output,        const TrackerGeometry * tracker)
{

    std::vector<TrajectoryMeasurement> measurements =traj_iterator->measurements();
    
    std::vector<TrajectoryMeasurement>::iterator traj_mes_iterator;
    //edm::LogInfo("TrackInfoProducer") << "Number of Measurements: "<<measurements.size();
    TrackInfo::TrajectoryInfo trajinfo;
    int nhit=0;
    for(traj_mes_iterator=measurements.begin();traj_mes_iterator!=measurements.end();++traj_mes_iterator){//loop on measurements
      
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
      TrackingRecHit const * thehitptr=nullptr;
      int i=0,j=0;

      for (thehit=track->recHitsBegin();thehit!=track->recHitsEnd();thehit++){
	i++;
	LocalPoint hitpos;
	if ((*thehit)->isValid())hitpos=(*thehit)->localPosition();
	if((*thehit)->geographicalId().rawId()==detid&&
	   (hitpos - pos).mag() < 1e-4)
	  {
	    thehitptr=(*thehit);
            thehitref = track->extra()->recHitRef(i-1);
	    j++;
	    break;
	  }
      }
      
      PTrajectoryStateOnDet const & fwdptsod=trajectoryStateTransform::persistentState( fwdtsos,detid);
      PTrajectoryStateOnDet const &  bwdptsod=trajectoryStateTransform::persistentState( bwdtsos,detid);
      PTrajectoryStateOnDet const &  updatedptsod=trajectoryStateTransform::persistentState( updatedtsos,detid);
      PTrajectoryStateOnDet const &  combinedptsod=trajectoryStateTransform::persistentState( combinedtsos,detid);
      

      const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>( thehitptr);
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>( thehitptr);

      RecHitType type=Single;
      LocalVector monofwd, stereofwd;
      LocalVector monobwd, stereobwd;
      LocalVector monoco, stereoco;
      LocalVector monoup, stereoup;

      LocalPoint pmonofwd, pstereofwd;
      LocalPoint pmonobwd, pstereobwd;
      LocalPoint pmonoco, pstereoco;
      LocalPoint pmonoup, pstereoup;
      if(matchedhit){
	type=Matched;
	const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tracker->idToDet(matchedhit->geographicalId()));
	
	GlobalVector gtrkdirfwd=gdet->toGlobal(fwdptsod.parameters().momentum());
	GlobalVector gtrkdirbwd=gdet->toGlobal(bwdptsod.parameters().momentum());
	GlobalVector gtrkdirup=gdet->toGlobal(updatedptsod.parameters().momentum());
	GlobalVector gtrkdirco=gdet->toGlobal(combinedptsod.parameters().momentum());
	

	
	const GeomDetUnit * monodet=gdet->monoDet();
	
	monofwd=monodet->toLocal(gtrkdirfwd);
	monobwd=monodet->toLocal(gtrkdirbwd);
	monoup=monodet->toLocal(gtrkdirup);
	monoco=monodet->toLocal(gtrkdirco);

	pmonofwd=project(gdet,monodet,fwdptsod.parameters().position(),monofwd);
	pmonobwd=project(gdet,monodet,bwdptsod.parameters().position(),monobwd);
	pmonoup=project(gdet,monodet,updatedptsod.parameters().position(),monoup);
	pmonoco=project(gdet,monodet,combinedptsod.parameters().position(),monoco);


	const GeomDetUnit * stereodet=gdet->stereoDet();
	
	stereofwd=stereodet->toLocal(gtrkdirfwd);
	stereobwd=stereodet->toLocal(gtrkdirbwd);
	stereoup=stereodet->toLocal(gtrkdirup);
	stereoco=stereodet->toLocal(gtrkdirco);

	pstereofwd=project(gdet,stereodet,fwdptsod.parameters().position(),stereofwd);
	pstereobwd=project(gdet,stereodet,bwdptsod.parameters().position(),stereobwd);
	pstereoup=project(gdet,stereodet,updatedptsod.parameters().position(),stereoup);
	pstereoco=project(gdet,stereodet,combinedptsod.parameters().position(),stereoco);


      }
      else if(phit){
	type=Projected;
	const GluedGeomDet * gdet=static_cast<const GluedGeomDet *>(tracker->idToDet(phit->geographicalId()));
	
	GlobalVector gtrkdirfwd=gdet->toGlobal(fwdptsod.parameters().momentum());
	GlobalVector gtrkdirbwd=gdet->toGlobal(bwdptsod.parameters().momentum());
	GlobalVector gtrkdirup=gdet->toGlobal(updatedptsod.parameters().momentum());
	GlobalVector gtrkdirco=gdet->toGlobal(combinedptsod.parameters().momentum());
	const SiStripRecHit2D&  originalhit=phit->originalHit();
	const GeomDetUnit * det;
	if(!StripSubdetector(originalhit.geographicalId().rawId()).stereo()){
	  det=gdet->monoDet();
	  monofwd= det->toLocal(gtrkdirfwd);
	  monobwd= det->toLocal(gtrkdirbwd);
	  monoup=  det->toLocal(gtrkdirup);
	  monoco=  det->toLocal(gtrkdirco);
	  pmonofwd=project(gdet,det,fwdptsod.parameters().position(),monofwd);
	  pmonobwd=project(gdet,det,bwdptsod.parameters().position(),monobwd);
	  pmonoup=project(gdet,det,updatedptsod.parameters().position(),monoup);
	  pmonoco=project(gdet,det,combinedptsod.parameters().position(),monoco);
	}
	else{
	  det=gdet->stereoDet();
	  stereofwd= det->toLocal(gtrkdirfwd);
	  stereobwd= det->toLocal(gtrkdirbwd);
	  stereoup=  det->toLocal(gtrkdirup);
	  stereoco=  det->toLocal(gtrkdirco);
	  pstereofwd=project(gdet,det,fwdptsod.parameters().position(),stereofwd);
	  pstereobwd=project(gdet,det,bwdptsod.parameters().position(),stereobwd);
	  pstereoup=project(gdet,det,updatedptsod.parameters().position(),stereoup);
	  pstereoco=project(gdet,det,combinedptsod.parameters().position(),stereoco);
	}
      }
      TrackingRecHitInfo::TrackingStates states;
      if(!forwardPredictedStateTag_.empty()) states.insert(std::make_pair(FwPredicted, TrackingStateInfo(std::make_pair(monofwd,stereofwd), std::make_pair(pmonofwd,pstereofwd), fwdptsod)));
      if(!backwardPredictedStateTag_.empty())states.insert(std::make_pair(BwPredicted, TrackingStateInfo(std::make_pair(monobwd,stereobwd), std::make_pair(pmonobwd,pstereobwd), bwdptsod)));
      if(!updatedStateTag_.empty())states.insert(std::make_pair(Updated, TrackingStateInfo(std::make_pair(monoup,stereoup), std::make_pair(pmonoup,pstereoup), updatedptsod)));
      if(!combinedStateTag_.empty())states.insert(std::make_pair(Combined, TrackingStateInfo(std::make_pair(monoco,stereoco), std::make_pair(pmonoco,pstereoco), combinedptsod)));
     
      TrackingRecHitInfo  tkRecHitInfo(type, states);
      
      
      
      if(j!=0){
	trajinfo.insert(std::make_pair(thehitref,tkRecHitInfo));
      }
      //      else  edm::LogInfo("TrackInfoProducer") << "RecHit not associated ";
    }
    //edm::LogInfo("TrackInfoProducer") << "Found "<<nhit<< " hits";
    //if(fwdtrajinfo.size()!=nhit) edm::LogInfo("TrackInfoProducer") << "Number of trackinfos  "<<fwdtrajinfo.size()<< " doesn't match!";
    output=TrackInfo((traj_iterator->seed()),trajinfo);
    
}

LocalPoint TrackInfoProducerAlgorithm::project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const
{
  
  GlobalPoint globalpoint=(det->surface()).toGlobal(position);
  
  // position of the initial and final point of the strip in glued local coordinates
  LocalPoint projposition=(projdet->surface()).toLocal(globalpoint);
  
  //correct the position with the track direction
  
  float scale=-projposition.z()/trackdirection.z();
  
  projposition+= scale*trackdirection;
  
  return projposition;
}
