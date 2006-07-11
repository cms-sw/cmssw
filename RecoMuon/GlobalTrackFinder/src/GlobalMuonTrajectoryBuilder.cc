/**
 *  Class: GlobalMuonTrajectoryBuilder
 *
 *  Description: 
 *
 *             MuonHitsOption: 0 - tracker only
 *                             1 - include all muon hits
 *                             2 - include only first muon hit(s)
 *                             3 - include only selected muon hits
 *                             4 - combined
 *
 *
 *  $Date: 2006/07/09 17:41:19 $
 *  $Revision: 1.3 $
 *
 *  Author :
 *  N. Neumeister            Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
 *  porting author:
 *  C. Liu                   Purdue University
 *
**/

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrajectoryBuilder.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
//#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonSeedCleaner.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonReFitter.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par) :
  theTrajectoryBuilder(0),
  theTrajectorySmoother(0),
  theTrajectoryCleaner(0),
  theRefitter(0) {

  theDirection = static_cast<ReconstructionDirection>(par.getParameter<int>("Direction"));
  theMuonHitsOption = par.getParameter<int>("MuonHitsOption");
  thePtCut = par.getParameter<double>("ptCut");
  theProbCut = par.getParameter<double>("Chi2ProbabilityCut");
  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");


  theTkTrackRef.clear();




}


//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() {

  delete theRefitter;
  delete theTrajectoryCleaner;
  delete theTrajectorySmoother;
  delete theTrajectoryBuilder;

}


//--------------
// Operations --
//--------------
std::vector<Trajectory> GlobalMuonTrajectoryBuilder::trajectories(const reco::Track& staTrack, const edm::Event& iEvent, const edm::EventSetup& iSetup)  {

  std::vector<Trajectory> result;
  
// get tracker TrackCollection from Event
// edm::Handle<reco::TrackCollection> allTrackerTracks;
// iEvent.getByLabel(theTkTrackLabel,allTrackerTracks);
  
// narrow down the TrackCollection by matching Eta-Phi Region
// chooseTrackerTracks(staTrack, tkTracks);

// choose a set of Tracks from the TrackCollection by TrackMatcher
// std::vector<reco::Track&> matchedResult =  match(staTrack, tkTracks);

// TC matchedTrajs;
  
// std::vector<reco::Track&> theTkTrackRef; set as private member

// for(std::vector<reco::Track&>::const_iterator tkt = matchedResult.begin();
//   tkt = matchedResult.end();tkt++) {
//   build Trajectories from the tkTracks
//   std::vector<Trajectory> matchedTraj = getTrackerTraj(*tkt);
//   if (matchedTraj.size()>0) {
//     matchedTrajs.push_back(matchedTraj.front());
//     theTkTrackRef.push_back(*tkt); 
//   }    
// } 
// 
//   build combined Trajectories with muon hits options
//   TC tjs = build(staTrack, matchedTrajs);

//   set theTkTrackRef during the build
 
  return result;
}

//build combined trajectory from sta Track and tracker RecHits, common for both options
std::vector<Trajectory> GlobalMuonTrajectoryBuilder::build(const reco::Track& staTrack, 
							   const std::vector<Trajectory>& tkTrajs) { 
  std::vector<Trajectory> result;
  
  //
  // check and select muon measurements and measure occupancy of muon stations
  //   
  std::vector<int> stationHits(4,0);
  edm::OwnVector< const TransientTrackingRecHit> muonRecHits1; // all muon rechits
  edm::OwnVector< const TransientTrackingRecHit> muonRecHits2; // only first muon rechits
  if ( theMuonHitsOption > 0 ) checkMuonHits(**staTrack, muonRecHits1, muonRecHits2, stationHits);
  
  //
  // add muon hits and refit/smooth trajectories
  //
  TC refittedResult;
  
  int position = 0; //used to set TkTrackRef
  if ( theMuonHitsOption > 0 ) {
    for ( TI it = tkTrajs.begin(); it != tkTrajs.end(); it++ ) {
      
      // cut on tracks with low momenta
      const GlobalVector& mom = (*it).lastMeasurement().updatedState().globalMomentum();
      if ( mom.mag() < 2.5 || mom.perp() < thePtCut ) continue;
      RecHitContainer trackerRecHits = (*it).recHits();
      if ( theDirection == insideOut ){
	//         std::reverse(trackerRecHits.begin(),trackerRecHits.end());
	edm::OwnVector< const TransientTrackingRecHit> temp; 
	edm::OwnVector< const TransientTrackingRecHit>::const_iterator rbegin = trackerRecHits.end();
	RecHitContainer::const_iterator rend = trackerRecHits.begin();
	rbegin--;
	rend--;
	for (edm::OwnVector< const TransientTrackingRecHit>::const_iterator rh = rbegin; rh != rend; rh--) 
	  temp.push_back(&*rh);
	
	trackerRecHits.clear();
	for (edm::OwnVector< const TransientTrackingRecHit>::const_iterator rh = temp.begin(); rh != temp.end(); rh++)
	  temp.push_back(&*rh);
	temp.clear();  
	
      }
      
      TrajectoryMeasurement firstTM = ( theDirection == outsideIn ) ? (*it).firstMeasurement() : (*it).lastMeasurement();
      TrajectoryStateOnSurface firstTsos = firstTM.updatedState();
      firstTsos.rescaleError(100.);
      
      TC refitted1,refitted2,refitted3;
      vector<Trajectory*> refit(4);
      const Trajectory* finalTrajectory = 0;
      
      // tracker only track
      refit[0] =const_cast<Trajectory*>(&(*it));                 
      
      RecHitContainer rechits(trackerRecHits);
      
      // full track with all muon hits
      if ( theMuonHitsOption == 1 || theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
	
	//          rechits.insert(rechits.end(), muonRecHits1.begin(), muonRecHits1.end() );
	for (RecHitContainer::const_iterator mrh = muonRecHits1.begin(); 
	     mrh != muonRecHits1.end(); mrh++) 
	  rechits.push_back(&*mrh); 
	
	edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "Number of hits: "<<rechits.size();
	refitted1 = theRefitter->trajectories((*it).seed(),rechits,firstTsos);
	if ( refitted1.size() == 1 ) {
	  refit[1] = &(*refitted1.begin());
	  if ( theMuonHitsOption == 1 ) finalTrajectory = &(*refitted1.begin());
	} else { 
	  if ( refitted1.size() == 0) theTkTrackRef.erase(theTkTrackRef.begin()+position); 
	}
	
      }
      
      // only first muon hits
      if ( theMuonHitsOption == 2 || theMuonHitsOption == 4 ) {
	rechits = trackerRecHits;
	//          rechits.insert(rechits.end(), muonRecHits2.begin(), muonRecHits2.end() );
	for (RecHitContainer::const_iterator mrh = muonRecHits1.begin();
	     mrh != muonRecHits1.end(); mrh++)
	  rechits.push_back(&*mrh);
	
	edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "Number of hits: "<<rechits.size();
	
	refitted2 = theRefitter->trajectories((*it).seed(),rechits,firstTsos);
	if ( refitted2.size() == 1 ) {
	  refit[2] = &(*refitted2.begin());
	  if ( theMuonHitsOption == 2 ) finalTrajectory = &(*refitted2.begin());
	}else {
	  if ( refitted2.size() == 0) theTkTrackRef.erase(theTkTrackRef.begin()+position);
	}
	
      } 
      
      // only selected muon hits
      if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
	RecHitContainer muonRecHits3;
	if ( refitted1.size() == 1 ) muonRecHits3 = selectMuonHits(*refitted1.begin(),stationHits);
	rechits = trackerRecHits;
	//          rechits.insert(rechits.end(), muonRecHits3.begin(), muonRecHits3.end() );
	for (RecHitContainer::const_iterator mrh = muonRecHits1.begin();
	     mrh != muonRecHits1.end(); mrh++)
	  rechits.push_back(&*mrh);
	
	edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "Number of hits: "<<rechits.size();
	
	refitted3 = theRefitter->trajectories((*it).seed(),rechits,firstTsos);
	if ( refitted3.size() == 1 ) {
	  refit[3] = &(*refitted3.begin());
	  if ( theMuonHitsOption == 3 ) finalTrajectory = &(*refitted3.begin());
	}else {
	  if ( refitted3.size() == 0) theTkTrackRef.erase(theTkTrackRef.begin()+position);
	}
	
      }
      
      if ( theMuonHitsOption == 4 ) {
	finalTrajectory = chooseTrajectory(refit);
	
      } 
      
      if ( finalTrajectory ) {
	refittedResult.push_back(*finalTrajectory);
      }
      position++;
    }
    
  }
  else {
    refittedResult = tkTrajs;
  }
  //    int nRefitted = refittedResult.size();
  
  //
  // muon trajectory cleaner
  //
  //    TrajectoryCleaner* mcleaner = new L3MuonTrajectoryCleaner();
  //    mcleaner->clean(refittedResult);
  //    delete mcleaner;
  
  //    if ( cout.testOut ) {
  //      cout.testOut << "seeds    : " << setw(3) << nSeeds << endl; 
  //      cout.testOut << "raw      : " << setw(3) << nRaw << endl;
  //      cout.testOut << "smoothed : " << setw(3) << nSmoothed << endl;
  //      cout.testOut << "cleaned  : " << setw(3) << nCleaned << endl;
  //      cout.testOut << "refitted : " << setw(3) << nRefitted << endl;
  //    }
  
  //
  // Perform a ghost suppression on all candidates, not only on those coming
  // from the same seed (RecMuon)
  //  FIXME
  result = refittedResult;
  return result;
  
  
}

/// return the TrackRef of tracker that is used in the final combined Trajectory
std::vector<reco::Track&> GlobalMuonTrajectoryBuilder::chosenTrackerTrackRef() const{
  return theTkTrackRef;
}

/// choose a set of Track that match given standalone Track by eta-phi region
std::vector<reco::Track&> GlobalMuonTrajectoryBuilder::chooseTrackerTracks(const reco::Track& staTrack, reco::TrackCollection& tkTracks) const{
  
  // define eta-phi region
  RectangularEtaPhiTrackingRegion regionOfInterest = defineRegionOfInterest(staTrack);
  
  std::vector<reco::Track&> tktracksInArea;
  
  vector<reco::Track>::const_iterator is;
  for ( is = tkTracks.begin(); is != tkTracks.end(); is++ ) {
    PTrajectoryStateOnDet ptsos = (*is).startingState();
    const GeomDet* gdet = theTrackingGeometry->idToDet(DetId(ptsos.detId()));
    TrajectoryStateOnSurface tsos = tsTransform.transientState(ptsos, &(gdet->surface()), &*theField);
    float eta = tsos.globalMomentum().eta();
    float phi = tsos.globalMomentum().phi();
    float deta(fabs(eta-regionOfInterest.direction().eta()));
    float dphi(fabs(Geom::Phi<float>(phi)-Geom::Phi<float>(regionOfInterest.direction().phi())));
    if ( deta > deltaEta || dphi > deltaPhi ) continue;  
    if ( deta > deltaEta || dphi > deltaPhi ) continue;     
    if ( nseeds < theMaxSeeds ) {
      tkTracksInArea.push_back(const_cast<TrajectorySeed*>(&((*is)))); 
      nseeds++;
    }
    else {
      break;
    } 
  }
  return tkTracksInArea;
}

/// get silicon tracker Trajectories from track Track and Seed directly
std::vector<Trajectory> GlobalMuonTrajectoryBuilder::getTrackerTraj(const reco::Track& tkTrack) const{

  TC result;

  //setES to get theFitter,thePropagator, TransientTrackingRecHitBuilder...

  //setEvent to get TrajectorySeeds in Tracker

  //use TransientTrackingRecHitBuilder to get TransientTrackingRecHits 
  //use TransientTrackBuilder to get a starting TSOS
  // TC  trjs = getTrackerTrajs (theFitter,thePropagator,hits,theTSOS,seeds);
  // result.insert(...); 
  
  return result;
}

std::vector<Trajectory> GlobalMuonTrajectoryBuilder::getTrackerTrajs (const TrajectoryFitter * theFitter,
								      const Propagator * thePropagator,
								      edm::OwnVector<const TransientTrackingRecHit>& hits,
								      TrajectoryStateOnSurface& theTSOS,
								      const TrajectorySeedCollection& seeds) const
{
  
  std::vector<Trajectory> result;
  
  for (TrajectorySeedCollection::const_iterator seed = seeds.begin();
       seed != seeds.end(); seed++) {
       //perform the fit: the result's size is 1 if it succeded, 0 if fails
       std::vector<Trajectory> trajs = theFitter->fit(*seed, hits, theTSOS);
       
       if (trajs.size() > 0) result.insert(result.end(),trajs.begin(),trajs.end());
  }
  
  edm::LogInfo("GlobalMuonTrajectoryBuilder")<<"FITTER FOUND "<<result.size()<<" TRAJECTORIES";
  return result;
  
}

void GlobalMuonTrajectoryBuilder::setES(const edm::EventSetup& setup,
				  edm::ESHandle<TrajectoryFitter>& theFitter,
				  edm::ESHandle<Propagator>& thePropagator)
{
  setup.get<IdealMagneticFieldRecord>().get(theField);  
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);
  //
  // get the fitter from the ES
  //
  std::string fitterName = "Fittername"; //FIXME
  setup.get<TrackingComponentsRecord>().get(fitterName,theFitter);
  //
  // get also the propagator
  //
  std::string propagatorName = "Propagatorname"; //FIXME   
  setup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  //
  // get the builder
  //
  std::string builderName = "TTRBname"; // FIXME  
  setup.get<TransientRecHitRecord>().get(builderName,theTransientHitBuilder);


}
//
//  check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
//
void GlobalMuonTrajectoryBuilder::checkMuonHits(const reco::Track& muon, 
                                          RecHitContainer& all,
                                          RecHitContainer& first,
                                          std::vector<int>& hits) const {
 
  int dethits[4];
  for ( int i=0; i<4; i++ ) hits[i]=dethits[i]=0;

  MuonDetLayerMeasurements theLayerMeasurements;

  RecHitContainer muonRecHits = getTransientHits(muon);

  // loop through all muon hits and calculate the maximum # of hits in each chamber      
  for (RecHitContainer::const_iterator imrh = muonRecHits.begin(); imrh != muonRecHits.end(); imrh++ ) { 

    if ( !(*imrh).isValid() ) continue;

    if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {

      int station = 0;
      int detRecHits = 0;
      const MuonTransientTrackingRecHit* immrh = dynamic_cast<const MuonTransientTrackingRecHit*>(&(*imrh));
      DetId id = immrh->geographicalId();

      const DetLayer* layer = theDetLayerGeometry->idToLayer(id);
      std::vector<MuonTransientTrackingRecHit*> dRecHits = theLayerMeasurements.recHits(layer);

      // get station of hit if it is in DT
      if ( (*immrh).isDT() ) {
        DTChamberId did(id.rawId());
        station = did.station();
        float coneSize = 10.0;

        bool hitUnique = false;
        RecHitContainer all2dRecHits;
        for (std::vector<MuonTransientTrackingRecHit*>::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
          if ( (**ir).dimension() == 2 ) {
            hitUnique = true;
            if ( all2dRecHits.size() > 0 ) {
              for (RecHitContainer::const_iterator iir = all2dRecHits.begin(); iir != all2dRecHits.end(); iir++ ) 
                 if (((*iir).localPosition()-(**ir).localPosition()).mag()<0.01) hitUnique = false;
            } //end of if ( all2dRecHits.size() > 0 )
            if ( hitUnique ) all2dRecHits.push_back(*ir);
          } else {
            RecHitContainer sRecHits = (**ir).transientHits();
            for (RecHitContainer::const_iterator iir = sRecHits.begin(); iir != sRecHits.end(); iir++ ) {
              if ( (*iir).dimension() == 2 ) {
                hitUnique = true;
                if ( all2dRecHits.size() > 0 ) {
                  for (RecHitContainer::const_iterator iiir = all2dRecHits.begin(); iiir != all2dRecHits.end(); iiir++ ) 
                      if (((*iiir).localPosition()-(*iir).localPosition()).mag()<0.01) hitUnique = false;
                }//end of if ( all2dRecHits.size() > 0 )
              }//end of if ( (*iir).dimension() == 2 ) 
              if ( hitUnique )
                   all2dRecHits.push_back(&*iir);
              break;
            }//end of for sRecHits
          }// end of else
      }// end of for loop over dRecHits
      for (RecHitContainer::const_iterator ir = all2dRecHits.begin(); ir != all2dRecHits.end(); ir++ ) {
        double rhitDistance = ((*ir).localPosition()-(*immrh).localPosition()).mag();
        if ( rhitDistance < coneSize ) detRecHits++;
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<<" Station "<<station<<" DT "<<(*ir).dimension()<<" " << (*ir).localPosition()
              <<" Distance: "<< rhitDistance<<" recHits: "<< detRecHits;
      }// end of for all2dRecHits
    }// end of if DT
    // get station of hit if it is in CSC
    else if ( (*immrh).isCSC() ) {
      CSCDetId did(id.rawId());
      station = did.station();

      float coneSize = 10.0;

      for (std::vector<MuonTransientTrackingRecHit*>::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
        double rhitDistance = ((**ir).localPosition()-(*immrh).localPosition()).mag();
        if ( rhitDistance < coneSize ) detRecHits++;
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<<" Station "<<station<< " CSC "<<(**ir).dimension()<<" "<<(**ir).localPosition()
                  <<" Distance: "<< rhitDistance<<" recHits: "<<detRecHits;
      }
    }
    // get station of hit if it is in RPC
    else if ( (*immrh).isRPC() ) {
      RPCDetId rpcid(id.rawId());
      station = rpcid.station();
      float coneSize = 100.0;
      for (std::vector<MuonTransientTrackingRecHit*>::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
        double rhitDistance = ((**ir).localPosition()-(*immrh).localPosition()).mag();
        if ( rhitDistance < coneSize ) detRecHits++;
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<<" Station "<<station<<" RPC "<<(**ir).dimension()<<" "<< (**ir).localPosition()
                  <<" Distance: "<<rhitDistance<<" recHits: "<<detRecHits;
      }
    }
    else {
      continue;
    }

    if ( (station > 0) && (station < 5) ) {
      int detHits = dRecHits.size();
      dethits[station-1] += detHits;
      if ( detRecHits > hits[station-1] ) hits[station-1] = detRecHits;
    }
    } //end of if option 3, 4
    all.push_back((&*imrh));
  } // end of loop over muon rechits
  if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 )  {
    for ( int i = 0; i < 4; i++ ) {
      edm::LogInfo("GlobalMuonTrajectoryBuilder")<<"Station "<<i+1<<": "<<hits[i]<<" "<<dethits[i]; 
    }
  }

  //
  // check order of muon measurements
  //
  if ((*all.begin()).globalPosition().mag() >
       (*(all.end()-1)).globalPosition().mag() ) {
     edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "reverse order: ";
     all.sort(RecHitLessByDet(alongMomentum));
  }


  int station1 = -999;
  int station2 = -999;

  for (RecHitContainer::const_iterator ihit = all.begin(); ihit != all.end(); ihit++ ) {
    if ( !(*ihit).isValid() ) continue;
    station1 = -999; station2 = -999;

    // store muon hits one at a time.
    first.push_back(&*ihit);
    
    const MuonTransientTrackingRecHit* immrh = dynamic_cast<const MuonTransientTrackingRecHit*>(&(*ihit));
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
    RecHitContainer::const_iterator nexthit(ihit);
    nexthit++;

    if ( ( nexthit != all.end()) && (*(nexthit)).isValid() ) {

      const MuonTransientTrackingRecHit* immrh2 = dynamic_cast<const MuonTransientTrackingRecHit*>(&(*nexthit));
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
          edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "checkMuonHits:";
          edm::LogInfo("GlobalMuonTrajectoryBuilder")<< " station 1 = "<<station1 
                        <<", r = "<< (*ihit).globalPosition().perp()
              	        <<", z = "<< (*ihit).globalPosition().z() << ", "; 
		        
          edm::LogInfo("GlobalMuonTrajectoryBuilder")<< " station 2 = " << station2
                        <<", r = "<<(*(nexthit)).globalPosition().perp()
                        <<", z = "<<(*(nexthit)).globalPosition().z() << ", ";
        return;
      }
    }
    else if ( (nexthit == all.end()) && (station1 != -999) ) {
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "checkMuonHits:";
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<< " station 1 = "<< station1
                      << ", r = " << (*ihit).globalPosition().perp()
                      << ", z = " << (*ihit).globalPosition().z() << ", "; 
      return;
    }
  }
  // if none of the above is satisfied, return blank vector.
  first.clear();

}


//
//  select muon hits compatible with trajectory; check hits in chambers with showers
//
edm::OwnVector<const TransientTrackingRecHit> GlobalMuonTrajectoryBuilder::selectMuonHits(const Trajectory& track, const std::vector<int>& hits) const {

  RecHitContainer muonRecHits;
 
  return muonRecHits;

}


//
// calculate the tail probability (-ln(P)) of a fit
//
double GlobalMuonTrajectoryBuilder::trackProbability(const Trajectory& track) const {

  int nDOF = 0;
  RecHitContainer rechits = track.recHits();
  for (RecHitContainer::const_iterator i = rechits.begin(); i != rechits.end(); ++i ) {
    if ((*i).isValid()) nDOF += (*i).dimension();
  }
  nDOF = max(nDOF - 5, 0);
  double prob = -LnChiSquaredProbability(track.chiSquared(), nDOF);

  return prob;

}

//
// print RecHits
//
void GlobalMuonTrajectoryBuilder::printHits(const RecHitContainer& hits) const {

  MuonPatternRecoDumper debug;
  edm::LogInfo("GlobalMuonTrajectoryBuilder")<<"Used RecHits : ";
  for (RecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir).isValid() ) {
      edm::LogInfo("GlobalMuonTrajectoryBuilder")<<"invalid recHit";
      continue; 
    }

    edm::LogInfo("GlobalMuonTrajectoryBuilder")<<"r = "
                  << sqrt((*ir).globalPosition().x() * (*ir).globalPosition().x() +
                         (*ir).globalPosition().y() * (*ir).globalPosition().y())
                  << "  z = " 
                  << (*ir).globalPosition().z()
                  << "  dimension = " << (*ir).dimension();
   debug.dumpMuonId(ir->geographicalId()); 
//                  << "  " << (*ir).det().detUnits().front()->type().module()
//                  << "  " << (*ir).det().detUnits().front()->type().part();
  }

}

edm::OwnVector<const TransientTrackingRecHit> GlobalMuonTrajectoryBuilder::getTransientHits(const reco::Track& track) const {
   RecHitContainer result;
   return result;

}

TrackingRegion GlobalMuonTrajectoryBuilder::defineRegionOfInterest(reco::Track& staTrack) {
  // track at innermost muon station
  TrajectoryStateOnSurface traj = staTrack.firstMeasurement().updatedState();
  if ( staTrack.direction() == oppositeToMomentum ) 
    traj = staTrack.lastMeasurement().updatedState();

  //FIXME: the result of STA should contain Trajectory or something more
  if ( !traj.isValid() ) return;

  // propagate to the outer tracker surface (r = 123.3cm, halfLength = 293.5cm)
  //
  GlobalPoint vertexPos(0.0,0.0,0.0);
  GlobalError vertexErr(0.0001,0.0,0.0001,0.0,0.0,28.09);
  MuonUpdatorAtVertex updator(vertexPos,vertexErr,&*theField);
  MuonVertexMeasurement vm = updator.update(traj);
  TrajectoryStateOnSurface traj_trak = vm.stateAtTracker();

  if ( !traj_trak.isValid() ) return;

  // rescale errors
  traj_trak.rescaleError(theErrorRescale);
  
  // define tracker region of interest
  GlobalVector mom = traj_trak.globalMomentum();

  TrajectoryStateOnSurface traj_vertex = vm.stateAtVertex();
  if ( traj_vertex.isValid() ) mom = traj_vertex.globalMomentum();
  float eta1   = mom.eta();
  float phi1   = mom.phi();
  float theta1 = mom.theta();

  float eta2 = 0.0;
  float phi2 = 0.0;
  float theta2 = 0.0;
  const edm::OwnVector< const TransientTrackingRecHit> recHits = staTrack.recHits();
  const TransientTrackingRecHit& r = *(recHits.begin()+1);
  eta2   = r.globalPosition().eta();
  phi2   = r.globalPosition().phi();
  theta2 = r.globalPosition().theta();
  
  float deta(fabs(eta1-eta2));
  float dphi(fabs(Geom::Phi<float>(phi1)-Geom::Phi<float>(phi2)));

  double deltaEta = 0.05;
  double deltaPhi = 0.07; // 5*ephi;
  double deltaZ   = min(15.9,3*sqrt(vertexErr.czz()));
  double minPt    = max(1.5,mom.perp()*0.6);
  
  Geom::Phi<float> phi(phi1);
  Geom::Theta<float> theta(theta1);
  if ( deta > 0.06 ) {
    deltaEta += (deta/2.);
  } 
  if ( dphi > 0.07 ) {
    deltaPhi += 0.15;
    if ( fabs(eta2) < 1.0 && mom.perp() < 6. ) deltaPhi = dphi;
  }
  if ( fabs(eta1) < 1.25 && fabs(eta1) > 0.8 ) deltaEta = max(0.07,deltaEta);
  if ( fabs(eta1) < 1.3  && fabs(eta1) > 1.0 ) deltaPhi = max(0.3,deltaPhi);

  GlobalVector direction(theta,phi,mom.perp());

  RectangularEtaPhiTrackingRegion rectRegion(direction, vertexPos,
                                             minPt, 0.2, deltaZ, deltaEta, deltaPhi);
  
  return rectRegion;
}
