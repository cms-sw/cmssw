/**
 *  Class: GlobalTrajectoryBuilderBase
 *
 *   Base class for GlobalMuonTrajectoryBuilder and L3MuonTrajectoryBuilder
 *   Provide common tools and interface to reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information).
 *   It tries to reconstruct the corresponding
 *   track in the tracker and performs
 *   matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *
 *  $Date: 2009/06/02 18:26:35 $
 *  $Revision: 1.34 $
 *  $Date: 2009/06/02 18:26:35 $
 *  $Revision: 1.34 $
 *
 *  \author N. Neumeister        Purdue University
 *  \author C. Liu               Purdue University
 *  \author A. Everett           Purdue University
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------
GlobalTrajectoryBuilderBase::GlobalTrajectoryBuilderBase(const edm::ParameterSet& par,
                                                         const MuonServiceProxy* service) : 
 theService(service) {

  theCategory = par.getUntrackedParameter<string>("Category", "Muon|RecoMuon|GlobalMuon|GlobalTrajectoryBuilderBase");

  theLayerMeasurements = new MuonDetLayerMeasurements(par.getParameter<InputTag>("DTRecSegmentLabel"),
						      par.getParameter<InputTag>("CSCRecSegmentLabel"),
						      par.getParameter<InputTag>("RPCRecSegmentLabel"));
  
  string MatcherOutPropagator = par.getParameter<string>("MatcherOutPropagator");
  string TransformerOutPropagator = par.getParameter<string>("TransformerOutPropagator");
  
  ParameterSet trackMatcherPSet = par.getParameter<ParameterSet>("GlobalMuonTrackMatcher");
  trackMatcherPSet.addParameter<string>("Propagator",MatcherOutPropagator);
  theTrackMatcher = new GlobalMuonTrackMatcher(trackMatcherPSet,theService);
  
  theTrackerPropagatorName = par.getParameter<string>("TrackerPropagator");

  ParameterSet trackTransformerPSet = par.getParameter<ParameterSet>("TrackTransformer");
  trackTransformerPSet.addParameter<string>("Propagator",TransformerOutPropagator);
  theTrackTransformer = new TrackTransformer(trackTransformerPSet);

  ParameterSet regionBuilderPSet = par.getParameter<ParameterSet>("MuonTrackingRegionBuilder");
  regionBuilderPSet.addParameter<bool>("RegionalSeedFlag",false);

  theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet,theService);

  theTrackerRecHitBuilderName = par.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = par.getParameter<string>("MuonRecHitBuilder");  

  theRPCInTheFit = par.getParameter<bool>("RefitRPCHits");

  theMuonHitsOption = par.getParameter<int>("MuonHitsOption");
  theTECxScale = par.getParameter<double>("ScaleTECxFactor");
  theTECyScale = par.getParameter<double>("ScaleTECyFactor");
  thePtCut = par.getParameter<double>("PtCut");
  theProbCut = par.getParameter<double>("Chi2ProbabilityCut");
  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");
  theKFFitterName = par.getParameter<string>("KFFitter");

  theCacheId_TRH = 0;

}


//--------------
// Destructor --
//--------------
GlobalTrajectoryBuilderBase::~GlobalTrajectoryBuilderBase() {

  if (theTrackMatcher) delete theTrackMatcher;
  if (theLayerMeasurements) delete theLayerMeasurements;
  if (theRegionBuilder) delete theRegionBuilder;
  if (theTrackTransformer) delete theTrackTransformer;

}


//
// set Event
//
void GlobalTrajectoryBuilderBase::setEvent(const edm::Event& event) {
  
  theEvent = &event;
  theLayerMeasurements->setEvent(event);  
  theService->eventSetup().get<TrajectoryFitter::Record>().get(theKFFitterName,theKFFitter);
  theTrackTransformer->setServices(theService->eventSetup());
  theRegionBuilder->setEvent(event);

  unsigned long long newCacheId_TRH = theService->eventSetup().get<TransientRecHitRecord>().cacheIdentifier();
  if ( newCacheId_TRH != theCacheId_TRH ) {
    LogDebug(theCategory) << "TransientRecHitRecord changed!";
    theCacheId_TRH = newCacheId_TRH;
    theService->eventSetup().get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    theService->eventSetup().get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);

  }

}


//
// build a combined tracker-muon trajectory
//
MuonCandidate::CandidateContainer 
GlobalTrajectoryBuilderBase::build(const TrackCand& staCand,
                                   MuonCandidate::CandidateContainer& tkTrajs) const {

  // MuonHitsOption: 0 - tracker only
  //                 1 - include all muon hits
  //                 2 - include only first muon hit(s)
  //                 3 - include only selected muon hits
  //                 4 - combined
  //                 5 - new combined
  //

  // tracker trajectory should be built and refit before this point
  LogTrace(theCategory) << "build begin. ";

  if ( tkTrajs.empty() ) return CandidateContainer();

  // add muon hits and refit/smooth trajectories
  CandidateContainer refittedResult;


  
  if ( theMuonHitsOption > 0 ) {

    vector<int> stationHits(4,0);
    ConstRecHitContainer muonRecHits0; // no muon hits
    ConstRecHitContainer muonRecHits1; // all muon rechits
    ConstRecHitContainer muonRecHits2; // only first muon rechits
    ConstRecHitContainer muonRecHits3; // selected muon rechits

    // check and select muon measurements and
    // measure occupancy in muon stations
    checkMuonHits(*(staCand.second), muonRecHits1, muonRecHits2, stationHits);
    
    for ( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); it++ ) {

      // cut on tracks with low momenta
      if(  (*it)->trackerTrack()->p() < 2.5 || (*it)->trackerTrack()->pt() < thePtCut  ) continue;

      ConstRecHitContainer trackerRecHits;
      if ((*it)->trackerTrack().isNonnull()) {
	LogDebug(theCategory);
	trackerRecHits = getTransientRecHits(*(*it)->trackerTrack());
      } else {
	LogDebug(theCategory)<<"NEED HITS FROM TRAJ";
	//trackerRecHits = (*it)->trackerTrajectory()->recHits();
      }

      // check for single TEC RecHits in trajectories in the overalp region
      if ( fabs((*it)->trackerTrack()->eta()) > 0.95 && fabs((*it)->trackerTrack()->eta()) < 1.15 && (*it)->trackerTrack()->pt() < 60 ) {
        if ( theTECxScale < 0 || theTECyScale < 0 )
	  trackerRecHits = selectTrackerHits(trackerRecHits);
	else
	  fixTEC(trackerRecHits,theTECxScale,theTECyScale);
      }
		  
      RefitDirection recHitDir = checkRecHitsOrdering(trackerRecHits);
      if ( recHitDir == outToIn ) reverse(trackerRecHits.begin(),trackerRecHits.end());

      reco::TransientTrack tTT((*it)->trackerTrack(),&*theService->magneticField(),theService->trackingGeometry());
      TrajectoryStateOnSurface innerTsos = tTT.innermostMeasurementState();

      edm::RefToBase<TrajectorySeed> tmpSeed;
      if((*it)->trackerTrack()->seedRef().isAvailable()) tmpSeed = (*it)->trackerTrack()->seedRef();

      if ( !innerTsos.isValid() ) {
	    LogTrace(theCategory) << "inner Trajectory State is invalid. ";
	    continue;
      }
      innerTsos.rescaleError(100.);
                  
      TC refitted0,refitted1,refitted2,refitted3;
      vector<Trajectory*> refit(4);
      MuonCandidate* finalTrajectory = 0;

      // tracker only track
      if ( ! ((*it)->trackerTrajectory() && (*it)->trackerTrajectory()->isValid()) ) refitted0 =  theTrackTransformer->transform((*it)->trackerTrack()) ;


      // full track with all muon hits
      if ( theMuonHitsOption == 1 || theMuonHitsOption == 3 || theMuonHitsOption == 4 ||  theMuonHitsOption == 5 ) {
        refitted1 = glbTrajectory(tmpSeed, trackerRecHits, muonRecHits1,innerTsos);
      }

      // only first muon hits
      if ( theMuonHitsOption == 2 || theMuonHitsOption == 4 || theMuonHitsOption == 5 ) {
        refitted2 = glbTrajectory(tmpSeed,trackerRecHits, muonRecHits2,innerTsos);
      }

      // only selected muon hits
      if ( (theMuonHitsOption == 3 || theMuonHitsOption == 4 || theMuonHitsOption == 5) && refitted1.size() == 1 ) {
        muonRecHits3 = selectMuonHits(*refitted1.begin(),stationHits);
        refitted3 = glbTrajectory(tmpSeed,trackerRecHits, muonRecHits3,innerTsos);
      }

      refit[0] = ( refitted0.empty() ) ? 0 : &(*refitted0.begin());
      refit[1] = ( refitted1.empty() ) ? 0 : &(*refitted1.begin());
      refit[2] = ( refitted2.empty() ) ? 0 : &(*refitted2.begin());
      refit[3] = ( refitted3.empty() ) ? 0 : &(*refitted3.begin());

      if((*it)->trackerTrajectory() && (*it)->trackerTrajectory()->isValid() ) refit[0] = (*it)->trackerTrajectory();

      if (refit[0]) refit[0]->setSeedRef(tmpSeed);

      Trajectory * refitTkTraj = refit[0];

      const Trajectory* chosenTrajectory = chooseTrajectory(refit, theMuonHitsOption);
      if (chosenTrajectory && refitTkTraj) {
	    Trajectory *tmpTrajectory = new Trajectory(*chosenTrajectory);
	    tmpTrajectory->setSeedRef(tmpSeed);
	    finalTrajectory = new MuonCandidate(tmpTrajectory, (*it)->muonTrack(), (*it)->trackerTrack(), new Trajectory(*refitTkTraj));
      } else {
	    LogWarning(theCategory)<<"could not choose a valid trajectory. skipping the muon. no final trajectory.";
      }

      if ( finalTrajectory ) {
        refittedResult.push_back(finalTrajectory);
      }
    }
  }
  else {
    LogTrace(theCategory)<<"theMuonHitsOption="<<theMuonHitsOption<<"\n"
                      <<tkTrajs.size()<<" total trajectories.";
    // do not just copy the collection over
	// we need to refit it for the smoother to work properly

	// refittedResult = tkTrajs;
	  
    // loop over tracker trajectories and refit them
    for ( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); it++ ) {
      Trajectory * refitTkTraj = ((*it)->trackerTrajectory() && (*it)->trackerTrajectory()->isValid()) ? (*it)->trackerTrajectory() : 0;
      TC refitted0;
      if( ! refitTkTraj) refitted0 = theTrackTransformer->transform((*it)->trackerTrack());
      if( ! refitted0.empty() ) {
	refitTkTraj = &(*refitted0.begin());
	refitTkTraj->setSeedRef((*it)->trackerTrack()->seedRef());
      }
      
      refittedResult.push_back(new MuonCandidate(new Trajectory(*refitTkTraj),(*it)->muonTrack(),(*it)->trackerTrack(), new Trajectory(*refitTkTraj) ));
    }
  }

  // choose the best global fit for this Standalone Muon based on the track probability
  CandidateContainer selectedResult;
  MuonCandidate* tmpCand = 0;
  if ( refittedResult.size() > 0 ) tmpCand = *(refittedResult.begin());
  double minProb = 9999;

  for (CandidateContainer::const_iterator iter=refittedResult.begin(); iter != refittedResult.end(); iter++) {
    double prob = trackProbability(*(*iter)->trajectory());
    if (prob < minProb) {
      minProb = prob;
      tmpCand = (*iter);
    }
  }

  if ( tmpCand )  selectedResult.push_back(new MuonCandidate(new Trajectory(*(tmpCand->trajectory())), tmpCand->muonTrack(), tmpCand->trackerTrack(), new Trajectory( *(tmpCand->trackerTrajectory()) ) ) );

  for (CandidateContainer::const_iterator it = refittedResult.begin(); it != refittedResult.end(); ++it) {
    if ( (*it)->trajectory() ) delete (*it)->trajectory();
    if ( (*it)->trackerTrajectory() ) delete (*it)->trackerTrajectory();
    if ( *it ) delete (*it);
  }
  refittedResult.clear();

  return selectedResult;

}


//
// select tracker tracks within a region of interest
//
vector<GlobalTrajectoryBuilderBase::TrackCand> 
GlobalTrajectoryBuilderBase::chooseRegionalTrackerTracks(const TrackCand& staCand, 
                                                         const vector<TrackCand>& tkTs) {
  
  // define eta-phi region
  RectangularEtaPhiTrackingRegion regionOfInterest = defineRegionOfInterest(staCand.second);
  
  // get region's etaRange and phiMargin
  PixelRecoRange<float> etaRange = regionOfInterest.etaRange();
  TkTrackingRegionsMargin<float> phiMargin = regionOfInterest.phiMargin();

  vector<TrackCand> result;

  double deltaR_max = 1.0;

  for ( vector<TrackCand>::const_iterator is = tkTs.begin(); is != tkTs.end(); ++is ) {
    // check if each trackCand is in region of interest
//    bool inEtaRange = etaRange.inside(is->second->eta());
//    bool inPhiRange = (fabs(Geom::Phi<float>(is->second->phi()) - Geom::Phi<float>(regionOfInterest.direction().phi())) < phiMargin.right() ) ? true : false ;

    double deltaR_tmp = deltaR(static_cast<double>(regionOfInterest.direction().eta()),
							   static_cast<double>(regionOfInterest.direction().phi()),
							   is->second->eta(), is->second->phi());

    // for each trackCand in region, add trajectory and add to result
    //if ( inEtaRange && inPhiRange ) {
    if (deltaR_tmp < deltaR_max) {
      TrackCand tmpCand = TrackCand(*is);
      result.push_back(tmpCand);
    }
  }

  return result; 

}


//
// define a region of interest within the tracker
//
RectangularEtaPhiTrackingRegion 
GlobalTrajectoryBuilderBase::defineRegionOfInterest(const reco::TrackRef& staTrack) const {

  RectangularEtaPhiTrackingRegion* region1 = theRegionBuilder->region(staTrack);
  
  TkTrackingRegionsMargin<float> etaMargin(fabs(region1->etaRange().min() - region1->etaRange().mean()),
					   fabs(region1->etaRange().max() - region1->etaRange().mean()));
  
  RectangularEtaPhiTrackingRegion region2(region1->direction(),
					  region1->origin(),
					  region1->ptMin(),
					  region1->originRBound(),
					  region1->originZBound(),
					  etaMargin,
					  region1->phiMargin());
  
  delete region1;
  return region2;
  
}


//
// check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
//
void GlobalTrajectoryBuilderBase::checkMuonHits(const reco::Track& muon, 
						ConstRecHitContainer& all,
						ConstRecHitContainer& first,
						std::vector<int>& hits) const {

  int dethits[4];
  for ( int i=0; i<4; i++ ) hits[i]=dethits[i]=0;
  
  TransientTrackingRecHit::ConstRecHitContainer muonRecHits = getTransientRecHits(muon);

//  all.assign(muonRecHits.begin(),muonRecHits.end()); //FIXME: should use this

  // loop through all muon hits and calculate the maximum # of hits in each chamber      
  for (ConstRecHitContainer::const_iterator imrh = muonRecHits.begin(); imrh != muonRecHits.end(); imrh++ ) {

    if ( (*imrh != 0 ) && !(*imrh)->isValid() ) continue;

    if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 || theMuonHitsOption == 5 ) {
      
      int station = 0;
      int detRecHits = 0;
      
      DetId id = (*imrh)->geographicalId();
      const DetLayer* layer = theService->detLayerGeometry()->idToLayer(id);
      MuonRecHitContainer dRecHits = theLayerMeasurements->recHits(layer);
      // get station of hit if it is in DT
      if ( id.subdetId() == MuonSubdetId::DT ) {
        DTChamberId did(id.rawId());
        station = did.station();
        float coneSize = 10.0;
	
        bool hitUnique = false;
        ConstRecHitContainer all2dRecHits;
        for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
          if ( (**ir).dimension() == 2 ) {
            hitUnique = true;
            if ( all2dRecHits.size() > 0 ) {
              for (ConstRecHitContainer::const_iterator iir = all2dRecHits.begin(); iir != all2dRecHits.end(); iir++ ) 
		if (((*iir)->localPosition()-(*ir)->localPosition()).mag()<0.01) hitUnique = false;
            } //end of if ( all2dRecHits.size() > 0 )
            if ( hitUnique ) all2dRecHits.push_back((*ir).get()); //FIXME!!
          } else {
            ConstRecHitContainer sRecHits = (**ir).transientHits();
            for (ConstRecHitContainer::const_iterator iir = sRecHits.begin(); iir != sRecHits.end(); iir++ ) {
              if ( (*iir)->dimension() == 2 ) {
                hitUnique = true;
                if ( !all2dRecHits.empty() ) {
                  for (ConstRecHitContainer::const_iterator iiir = all2dRecHits.begin(); iiir != all2dRecHits.end(); iiir++ ) 
		    if (((*iiir)->localPosition()-(*iir)->localPosition()).mag()<0.01) hitUnique = false;
                }//end of if ( all2dRecHits.size() > 0 )
              }//end of if ( (*iir).dimension() == 2 ) 
              if ( hitUnique )
		all2dRecHits.push_back(*iir);
              break;
            }//end of for sRecHits
          }// end of else
	}// end of for loop over dRecHits
	for (ConstRecHitContainer::const_iterator ir = all2dRecHits.begin(); ir != all2dRecHits.end(); ir++ ) {
	  double rhitDistance = ((*ir)->localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
	  LogTrace(theCategory) << " Station " << station << " DT "<<(*ir)->dimension()<<" " << (*ir)->localPosition()
						      << " Distance: "<< rhitDistance<<" recHits: "<< detRecHits;
	}// end of for all2dRecHits
      }// end of if DT
      // get station of hit if it is in CSC
      else if ( id.subdetId() == MuonSubdetId::CSC ) {
	CSCDetId did(id.rawId());
	station = did.station();
	
	float coneSize = 10.0;
	
	for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
	  double rhitDistance = ((**ir).localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
	  LogTrace(theCategory) << " Station " << station << " CSC "<<(**ir).dimension()<<" "<<(**ir).localPosition()
                                                  << " Distance: "<< rhitDistance<<" recHits: "<<detRecHits;
	}
      }
      // get station of hit if it is in RPC
      else if ( id.subdetId() == MuonSubdetId::RPC ) {
	RPCDetId rpcid(id.rawId());
	station = rpcid.station();
	float coneSize = 100.0;
	for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
	  double rhitDistance = ((**ir).localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
	  LogTrace(theCategory)<<" Station "<<station<<" RPC "<<(**ir).dimension()<<" "<< (**ir).localPosition()
						     <<" Distance: "<<rhitDistance<<" recHits: "<<detRecHits;
	}
      }
      else {
        LogError(theCategory)<<" Wrong Hit Type ";
	continue;      
      }
      
      if ( (station > 0) && (station < 5) ) {
	int detHits = dRecHits.size();
	dethits[station-1] += detHits;
	if ( detRecHits > hits[station-1] ) hits[station-1] = detRecHits;
      }
    } //end of if option 3, 4, 5

    all.push_back((*imrh).get()); //FIXME: may need fast assignment on above

  } // end of loop over muon rechits
  if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 || theMuonHitsOption == 5 )  {
    for ( int i = 0; i < 4; i++ ) {
      LogTrace(theCategory) <<"Station "<<i+1<<": "<<hits[i]<<" "<<dethits[i]; 
    }
  }     
  
  //
  // check order of muon measurements
  //
  LogTrace(theCategory) << "CheckMuonHits "<<all.size();

  if ( (all.size() > 1) &&
       ( all.front()->globalPosition().mag() >
	 all.back()->globalPosition().mag() ) ) {
    LogTrace(theCategory)<< "reverse order: ";
    stable_sort(all.begin(),all.end(),RecHitLessByDet(alongMomentum));
  }

  stable_sort(all.begin(),all.end(),ComparatorInOut());
  
  int station1 = -999;
  int station2 = -999;
  for (ConstRecHitContainer::const_iterator ihit = all.begin(); ihit != all.end(); ihit++ ) {

    if ( !(*ihit)->isValid() ) continue;
    station1 = -999; station2 = -999;
    // store muon hits one at a time.
    first.push_back(*ihit);
    
    ConstMuonRecHitPointer immrh = dynamic_cast<const MuonTransientTrackingRecHit*>((*ihit).get()); //FIXME
    DetId id = immrh->geographicalId();
    
    // get station of 1st hit if it is in DT
    if ( (*immrh).isDT()  ) {
      DTChamberId did(id.rawId());
      station1 = did.station();
    }
    // otherwise get station of 1st hit if it is in CSC
    else if  ( (*immrh).isCSC() ) {
      CSCDetId did(id.rawId());
      station1 = did.station();
    }
    // check next RecHit
    ConstRecHitContainer::const_iterator nexthit(ihit);
    nexthit++;
    
    if ( ( nexthit != all.end()) && (*nexthit)->isValid() ) {
      
      ConstMuonRecHitPointer immrh2 = dynamic_cast<const MuonTransientTrackingRecHit*>((*nexthit).get());
      DetId id2 = immrh2->geographicalId();
      
      // get station of 1st hit if it is in DT
      if ( (*immrh2).isDT()  ) {
        DTChamberId did2(id2.rawId());
        station2 = did2.station();
      }
      // otherwise get station of 1st hit if it is in CSC
      else if  ( (*immrh2).isCSC() ) {
        CSCDetId did2(id2.rawId());
        station2 = did2.station();
      }
      
      // 1st hit is in station 1 and second hit is in a different station
      // or an rpc (if station = -999 it could be an rpc hit)
      if ( (station1 != -999) && ((station2 == -999) || (station2 > station1)) ) {
	LogTrace(theCategory) << "checkMuonHits:";
	LogTrace(theCategory) << " station 1 = "<<station1 
						   <<", r = "<< (*ihit)->globalPosition().perp()
						   <<", z = "<< (*ihit)->globalPosition().z() << ", "; 
	
	LogTrace(theCategory) << " station 2 = " << station2
						   <<", r = "<<(*(nexthit))->globalPosition().perp()
						   <<", z = "<<(*(nexthit))->globalPosition().z() << ", ";
	return;
      }
    }
    else if ( (nexthit == all.end()) && (station1 != -999) ) {
      LogTrace(theCategory) << "checkMuonHits:";
      LogTrace(theCategory) << " station 1 = "<< station1
                                              << ", r = " << (*ihit)->globalPosition().perp()
                                              << ", z = " << (*ihit)->globalPosition().z() << ", "; 
      return;
    }
  }
  // if none of the above is satisfied, return blank vector
  first.clear();

  return; 

}


//
// select muon hits compatible with trajectory; 
// check hits in chambers with showers
//
GlobalTrajectoryBuilderBase::ConstRecHitContainer 
GlobalTrajectoryBuilderBase::selectMuonHits(const Trajectory& traj, 
                                            const std::vector<int>& hits) const {

  ConstRecHitContainer muonRecHits;
  const double globalChi2Cut = 200.0;

  vector<TrajectoryMeasurement> muonMeasurements = traj.measurements(); 

  // loop through all muon hits and skip hits with bad chi2 in chambers with high occupancy      
  for (vector<TrajectoryMeasurement>::const_iterator im = muonMeasurements.begin(); im != muonMeasurements.end(); im++ ) {

    if ( !(*im).recHit()->isValid() ) continue;
    if ( (*im).recHit()->det()->geographicalId().det() != DetId::Muon ) continue;
    ConstMuonRecHitPointer immrh = dynamic_cast<const MuonTransientTrackingRecHit*>((*im).recHit().get());

    DetId id = immrh->geographicalId();
    int station = 0;
    int threshold = 0;
    double chi2Cut = 0.0;

    // get station of hit if it is in DT
    if ( (*immrh).isDT() ) {
      DTChamberId did(id.rawId());
      station = did.station();
      threshold = theHitThreshold;
      chi2Cut = theDTChi2Cut;
    }
    // get station of hit if it is in CSC
    else if ( (*immrh).isCSC() ) {
      CSCDetId did(id.rawId());
      station = did.station();
      threshold = theHitThreshold;
      chi2Cut = theCSCChi2Cut;
    }
    // get station of hit if it is in RPC
    else if ( (*immrh).isRPC() ) {
      RPCDetId rpcid(id.rawId());
      station = rpcid.station();
      threshold = theHitThreshold;
      chi2Cut = theRPCChi2Cut;
    }
    else {
      continue;
    }

    double chi2ndf = (*im).estimate()/(*im).recHit()->dimension();  

    bool keep = true;
    if ( (station > 0) && (station < 5) ) {
      if ( hits[station-1] > threshold ) keep = false;
    }   
    
    if ( (keep || ( chi2ndf < chi2Cut )) && ( chi2ndf < globalChi2Cut ) ) {
      muonRecHits.push_back((*im).recHit());
    } else {
      LogTrace(theCategory)
	<< "Skip hit: " << id.det() << " " << station << ", " 
	<< chi2ndf << " (" << chi2Cut << " chi2 threshold) " 
	<< hits[station-1] << endl;
    }

  }
  
  // check order of rechits
  reverse(muonRecHits.begin(),muonRecHits.end());

  return muonRecHits;

}


//
// choose final trajectory
//
const Trajectory* 
GlobalTrajectoryBuilderBase::chooseTrajectory(const std::vector<Trajectory*>& t, 
                                              int muonHitsOption) const {

  Trajectory* result = 0;
  
  if ( muonHitsOption == 0 ) {
    if (t[0]) result = t[0];
    return result;
  } else if ( muonHitsOption == 1 ) {
    if (t[1]) result = t[1];
    return result;
  } else if ( muonHitsOption == 2 ) {
    if (t[2]) result = t[2];
    return result;
  } else if ( muonHitsOption == 3 ) {
    if (t[3]) result = t[3];
    return result;
  } else if ( muonHitsOption == 4 ) {
    double prob0 = ( t[0] ) ? trackProbability(*t[0]) : 0.0;
    double prob1 = ( t[1] ) ? trackProbability(*t[1]) : 0.0;
    double prob2 = ( t[2] ) ? trackProbability(*t[2]) : 0.0;
    double prob3 = ( t[3] ) ? trackProbability(*t[3]) : 0.0; 
    
    LogTrace(theCategory) << "Probabilities: " << prob0 << " " << prob1 << " " << prob2 << " " << prob3 << endl;
    
    if ( t[1] ) result = t[1];
    if ( (t[1] == 0) && t[3] ) result = t[3];
  
    if ( t[1] && t[3] && ( (prob1 - prob3) > 0.05 )  )  result = t[3];

    if ( t[0] && t[2] && fabs(prob2 - prob0) > theProbCut ) {
      LogTrace(theCategory) << "select Tracker only: -log(prob) = " << prob0 << endl;
      result = t[0];
      return result;
    }

    if ( (t[1] == 0) && (t[3] == 0) && t[2] ) result = t[2];

    Trajectory* tmin = 0;
    double probmin = 0.0;
    if ( t[1] && t[3] ) {
      probmin = prob3; tmin = t[3];
      if ( prob1 < prob3 ) { probmin = prob1; tmin = t[1]; }
    }
    else if ( (t[3] == 0) && t[1] ) { 
      probmin = prob1; tmin = t[1]; 
    }
    else if ( (t[1] == 0) && t[3] ) {
      probmin = prob3; tmin = t[3]; 
    }

    if ( tmin && t[2] && ( (probmin - prob2) > 3.5 )  ) {
      result = t[2];
    }

  } else if ( muonHitsOption == 5 ) {

    double prob[4];
    int chosen=3;
    for (int i=0;i<4;i++) 
      prob[i] = (t[i]) ? trackProbability(*t[i]) : 0.0; 

    if (!t[3])
      if (t[2]) chosen=2; else
        if (t[1]) chosen=1; else
          if (t[0]) chosen=0;

    if ( t[0] && t[3] && ((prob[3]-prob[0]) > 48.) ) chosen=0;
    if ( t[0] && t[1] && ((prob[1]-prob[0]) < 3.) ) chosen=1;
    if ( t[2] && ((prob[chosen]-prob[2]) > 9.) ) chosen=2;
    
    LogTrace(theCategory) << "Chosen Trajectory " << chosen;
    
    result=t[chosen];
  }
  else {
    LogTrace(theCategory) << "Wrong Hits Option in Choosing Trajectory ";
  } 
  return result;

}


//
// calculate the tail probability (-ln(P)) of a fit
//
double 
GlobalTrajectoryBuilderBase::trackProbability(const Trajectory& track) const {

  if ( track.ndof() > 0 && track.chiSquared() > 0 ) { 
    return -LnChiSquaredProbability(track.chiSquared(), track.ndof());
  } else {
    return 0.0;
  }

}


//
// print RecHits
//
void GlobalTrajectoryBuilderBase::printHits(const ConstRecHitContainer& hits) const {

  LogTrace(theCategory) << "Used RecHits: " << hits.size();
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogTrace(theCategory) << "invalid RecHit";
      continue; 
    }
    
    const GlobalPoint& pos = (*ir)->globalPosition();
    
    LogTrace(theCategory) 
      << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y())
      << "  z = " << pos.z()
      << "  dimension = " << (*ir)->dimension()
      << "  " << (*ir)->det()->geographicalId().det()
      << "  " << (*ir)->det()->subDetector();

  }

}

//
// check order of RechIts on a trajectory
//
GlobalTrajectoryBuilderBase::RefitDirection
GlobalTrajectoryBuilderBase::checkRecHitsOrdering(const TransientTrackingRecHit::ConstRecHitContainer& recHits) const {

  if ( !recHits.empty() ) {
    ConstRecHitContainer::const_iterator frontHit = recHits.begin();
    ConstRecHitContainer::const_iterator backHit  = recHits.end() - 1;
    while ( !(*frontHit)->isValid() && frontHit != backHit ) {frontHit++;}
    while ( !(*backHit)->isValid() && backHit != frontHit )  {backHit--;}

    double rFirst = (*frontHit)->globalPosition().mag();
    double rLast  = (*backHit) ->globalPosition().mag();

    if ( rFirst < rLast ) return inToOut;
    else if (rFirst > rLast) return outToIn;
    else {
      LogError(theCategory) << "Impossible to determine the rechits order" << endl;
      return undetermined;
    }
  }
  else {
    LogError(theCategory) << "Impossible to determine the rechits order" << endl;
    return undetermined;
  }
}


//
//  build a global trajectory from tracker and muon hits
//
vector<Trajectory> 
GlobalTrajectoryBuilderBase::glbTrajectory(const edm::RefToBase<TrajectorySeed>& seed,
                                           const ConstRecHitContainer& tkhits,
                                           const ConstRecHitContainer& muonhits,
                                           const TrajectoryStateOnSurface& firstPredTsos) const {

  ConstRecHitContainer hits = tkhits;
  hits.insert(hits.end(), muonhits.begin(), muonhits.end());

  if ( hits.empty() ) return vector<Trajectory>();
  LogDebug(theCategory) << seed.isNull();
  PTrajectoryStateOnDet PTSOD = ( !seed.isNull() ) ? seed->startingState() : PTrajectoryStateOnDet();

  edm::OwnVector<TrackingRecHit> garbage2;

  RefitDirection recHitDir = checkRecHitsOrdering(hits);
  PropagationDirection refitDir = (recHitDir == outToIn) ? oppositeToMomentum : alongMomentum ;
  TrajectorySeed newSeed(PTSOD,garbage2,refitDir);

  TrajectoryStateOnSurface firstTsos = firstPredTsos;
  firstTsos.rescaleError(10.);

  vector<Trajectory> theTrajs = theKFFitter->fit(newSeed,hits,firstTsos);

  return theTrajs;

}


//
// select trajectories with only a single TEC hit
//
GlobalTrajectoryBuilderBase::ConstRecHitContainer 
GlobalTrajectoryBuilderBase::selectTrackerHits(const ConstRecHitContainer& all) const {
 
  int nTEC(0);

  ConstRecHitContainer hits;
  for (ConstRecHitContainer::const_iterator i = all.begin(); i != all.end(); i++) {
    if ( !(*i)->isValid() ) continue;
    if ( (*i)->det()->geographicalId().det() == DetId::Tracker &&
         (*i)->det()->geographicalId().subdetId() == StripSubdetector::TEC) {
      nTEC++;
    } else {
      hits.push_back((*i).get());
    }
    if ( nTEC > 1 ) return all;
  }

  return hits;

}


//
// rescale errors of outermost TEC RecHit
//
void GlobalTrajectoryBuilderBase::fixTEC(ConstRecHitContainer& all,
                                         double scl_x,
                                         double scl_y) const {

  int nTEC(0);
  ConstRecHitContainer::iterator lone_tec;

  for ( ConstRecHitContainer::iterator i = all.begin(); i != all.end(); i++) {
    if ( !(*i)->isValid() ) continue;
    
    if ( (*i)->det()->geographicalId().det() == DetId::Tracker &&
         (*i)->det()->geographicalId().subdetId() == StripSubdetector::TEC) {
      lone_tec = i;
      nTEC++;
      
      if ( (i+1) != all.end() && (*(i+1))->isValid() &&
          (*(i+1))->det()->geographicalId().det() == DetId::Tracker &&
          (*(i+1))->det()->geographicalId().subdetId() == StripSubdetector::TEC) {
        nTEC++;
        break;
      }
    }
    
    if (nTEC > 1) break;
  }
  
  if ( nTEC == 1 && (*lone_tec)->hit()->isValid() &&
       (*lone_tec)->hit()->geographicalId().det() == DetId::Tracker &&
       (*lone_tec)->hit()->geographicalId().subdetId() == StripSubdetector::TEC) {
    
    // rescale the TEC rechit error matrix in its rotated frame
    const SiStripRecHit2D* strip = dynamic_cast<const SiStripRecHit2D*>((*lone_tec)->hit());
    const TSiStripRecHit2DLocalPos* Tstrip = dynamic_cast<const TSiStripRecHit2DLocalPos*>((*lone_tec).get());
    if (strip && Tstrip->det() && Tstrip) {
      LocalPoint pos = Tstrip->localPosition();
      if ((*lone_tec)->detUnit()) {
        const StripTopology* topology = dynamic_cast<const StripTopology*>(&(*lone_tec)->detUnit()->topology());
        if (topology) {
          // rescale the local error along/perp the strip by a factor
          float angle = topology->stripAngle(topology->strip((*lone_tec)->hit()->localPosition()));
          LocalError error = Tstrip->localPositionError();
          LocalError rotError = error.rotate(angle);
          LocalError scaledError(rotError.xx() * scl_x * scl_x, 0, rotError.yy() * scl_y * scl_y);
          error = scaledError.rotate(-angle);
          MuonTransientTrackingRecHit* mtt_rechit;
          if (strip->cluster().isNonnull()) {
            //// the implemetantion below works with cloning
            //// to get a RecHitPointer to SiStripRecHit2D, the only  method that works is
            //// RecHitPointer MuonTransientTrackingRecHit::build(const GeomDet*,const TrackingRecHit*)
            SiStripRecHit2D* st = new SiStripRecHit2D(pos,error,
                                                      (*lone_tec)->geographicalId().rawId(),
                                                      strip->cluster());
            *lone_tec = mtt_rechit->build((*lone_tec)->det(),st);
          }
          else {
            SiStripRecHit2D* st = new SiStripRecHit2D(pos,error,
                                                      (*lone_tec)->geographicalId().rawId(),
                                                      strip->cluster_regional());
            *lone_tec = mtt_rechit->build((*lone_tec)->det(),st);
          }
        }
      }
    }
  }

}


//
// get transient RecHits
//
TransientTrackingRecHit::ConstRecHitContainer
GlobalTrajectoryBuilderBase::getTransientRecHits(const reco::Track& track) const {

  TransientTrackingRecHit::ConstRecHitContainer result;
  
  TrajectoryStateTransform tsTrans;

  TrajectoryStateOnSurface currTsos = tsTrans.innerStateOnSurface(track, *theService->trackingGeometry(), &*theService->magneticField());

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if((*hit)->isValid()) {
      DetId recoid = (*hit)->geographicalId();
      if ( recoid.det() == DetId::Tracker ) {
	TransientTrackingRecHit::RecHitPointer ttrhit = theTrackerRecHitBuilder->build(&**hit);
	TrajectoryStateOnSurface predTsos =  theService->propagator(theTrackerPropagatorName)->propagate(currTsos, theService->trackingGeometry()->idToDet(recoid)->surface());
	LogTrace(theCategory)<<"predtsos "<<predTsos.isValid();
	if ( predTsos.isValid() ) currTsos = predTsos;
	TransientTrackingRecHit::RecHitPointer preciseHit = ttrhit->clone(predTsos);
	result.push_back(preciseHit);
      } else if ( recoid.det() == DetId::Muon ) {
	if ( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit) {
	  LogDebug(theCategory) << "RPC Rec Hit discarded"; 
	  continue;
	}
	result.push_back(theMuonRecHitBuilder->build(&**hit));
      }
    }
  }
  
  return result;

  }
