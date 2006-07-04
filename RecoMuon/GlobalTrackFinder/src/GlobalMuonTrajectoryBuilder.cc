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
 *  $Date: 2006/06/18 19:14:36 $
 *  $Revision: 1.1 $
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

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

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
std::vector<Trajectory> GlobalMuonTrajectoryBuilder::trajectories(const reco::TrackRef* staTrack, const edm::Event& iEvent, const edm::EventSetup& iSetup)  {

   std::vector<Trajectory> result;

// get tracker TrackCollection from Event
//  edm::Handle<reco::TrackCollection> allTrackerTracks;
//  iEvent.getByLabel(theTkTrackLabel,allTrackerTracks);

// narrow down the TrackCollection by matching Eta-Phi Region
// chooseTrackerTracks(staTrack, tkTracks);

// choose a set of Tracks from the TrackCollection by TrackMatcher
// std::vector<reco::TrackRef*> matchedResult =  match(staTrack, tkTracks);

// TC matchedTrajs;
  
// std::vector<reco::TrackRef*> theTkTrackRef; set as private member

// for(std::vector<reco::TrackRef*>::const_iterator tkt = matchedResult.begin();
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
std::vector<Trajectory> GlobalMuonTrajectoryBuilder::build(const reco::TrackRef* staTrack, 
                                      const std::vector<Trajectory>& tkTrajs) { 
    std::vector<Trajectory> result;

    //
    // check and select muon measurements and measure occupancy of muon stations
    //   
    std::vector<int> stationHits(4,0);
    edm::OwnVector< const TransientTrackingRecHit> muonRecHits1; // all muon rechits
    edm::OwnVector< const TransientTrackingRecHit> muonRecHits2; // only first muon rechits
    if ( theMuonHitsOption > 0 ) checkMuonHits(*staTrack, muonRecHits1, muonRecHits2, stationHits);
   
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
          //reverse(trackerRecHits.begin(),trackerRecHits.end());
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
std::vector<reco::TrackRef*> GlobalMuonTrajectoryBuilder::chosenTrackerTrackRef() const{
  return theTkTrackRef;
}

/// choose a set of Track that match given standalone Track by eta-phi region
void GlobalMuonTrajectoryBuilder::chooseTrackerTracks(const reco::TrackRef* staTrack, const reco::TrackCollection& tkTracks) const{

}

/// get silicon tracker Trajectories from track Track and Seed directly
std::vector<Trajectory> GlobalMuonTrajectoryBuilder::getTrackerTraj(const reco::TrackRef* tkTrack) const{

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
				  edm::ESHandle<TrackerGeometry>& theG,
				  edm::ESHandle<MagneticField>& theMF,
				  edm::ESHandle<TrajectoryFitter>& theFitter,
				  edm::ESHandle<Propagator>& thePropagator,
				  edm::ESHandle<TransientTrackingRecHitBuilder>& theBuilder)
{
  //
  //get geometry
  //
  LogDebug("TrackProducer") << "get geometry" << "\n";
  setup.get<TrackerDigiGeometryRecord>().get(theG);
  //
  //get magnetic field
  //
  LogDebug("TrackProducer") << "get magnetic field" << "\n";
  setup.get<IdealMagneticFieldRecord>().get(theMF);  
  //
  // get the fitter from the ES
  //
  LogDebug("TrackProducer") << "get the fitter from the ES" << "\n";
  std::string fitterName = "Fittername"; //FIXME
  setup.get<TrackingComponentsRecord>().get(fitterName,theFitter);
  //
  // get also the propagator
  //
  LogDebug("TrackProducer") << "get also the propagator" << "\n";
  std::string propagatorName = "Propagatorname"; //FIXME   
  setup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  //
  // get the builder
  //
  LogDebug("TrackProducer") << "get also the TransientTrackingRecHitBuilder" << "\n";
  std::string builderName = "TTRBname"; // FIXME  
  setup.get<TransientRecHitRecord>().get(builderName,theBuilder);

}
//
//  check muon RecHits, calculate chamber occupancy and select hits to be used in the final fit
//
void GlobalMuonTrajectoryBuilder::checkMuonHits(const reco::TrackRef& muon, 
                                          RecHitContainer& all,
                                          RecHitContainer& first,
                                          std::vector<int>& hits) const {
 

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

