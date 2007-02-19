#include "AnalysisAlgos/TrackInfoProducer/interface/TrackInfoProducerAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Utilities/General/interface/CMSexception.h"


void TrackInfoProducerAlgorithm::run(std::vector<Trajectory>::const_iterator  traj_iterator,edm::Handle<TrackingRecHitCollection> *rechits,
				     reco::TrackInfo &outputFwd,
				     reco::TrackInfo &outputBwd,
				     reco::TrackInfo &outputUpdated, 
				     reco::TrackInfo &outputCombined)
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
      if (!ttrh->isValid()) continue;
      nhit++;
      unsigned int detid=ttrh->hit()->geographicalId().rawId();
      
      LocalPoint pos=ttrh->hit()->localPosition();
      TrackingRecHitCollection::const_iterator thehit;
      TrackingRecHitRef thehitref;
      int i=0,j=0;
      //edm::LogInfo("TrackInfoProducer") <<"Rechit size: "<<rechits->product()->size();
      for (thehit=rechits->product()->begin();thehit!=rechits->product()->end();thehit++){
	i++;
	if (!thehit->isValid()) continue;
	if(thehit->geographicalId().rawId()==detid&&
	   (thehit->localPosition() - pos).mag() < 1e-4)
	  {
	    thehitref=TrackingRecHitRef(*rechits,i-1);
	    //	  edm::LogInfo("TrackInfoProducer") << "Found a rechit ";
	    j++;
	    break;
	  }
      }
      TrajectoryStateTransform tsostransform;
      const PTrajectoryStateOnDet* fwdptsod=tsostransform.persistentState( fwdtsos,detid);
      const PTrajectoryStateOnDet* bwdptsod=tsostransform.persistentState( bwdtsos,detid);
      const PTrajectoryStateOnDet* updatedptsod=tsostransform.persistentState( updatedtsos,detid);
      const PTrajectoryStateOnDet* combinedptsod=tsostransform.persistentState( combinedtsos,detid);
      if(j!=0){
	fwdtrajinfo.insert(make_pair(thehitref,*fwdptsod));
	bwdtrajinfo.insert(make_pair(thehitref,*bwdptsod));
	updatedtrajinfo.insert(make_pair(thehitref,*updatedptsod));
	combinedtrajinfo.insert(make_pair(thehitref,*combinedptsod));
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
