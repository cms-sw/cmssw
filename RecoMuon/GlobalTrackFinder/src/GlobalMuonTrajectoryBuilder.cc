/**
 *  Class: GlobalMuonTrajectoryBuilder
 *
 *  Description:
 *   Reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information).
 *   It tries to reconstruct the corresponding
 *   track in the tracker and performs
 *   matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *
 *  $Date: 2007/01/17 16:18:04 $
 *  $Revision: 1.69 $
 *
 *  Authors :
 *  N. Neumeister            Purdue University
 *  C. Liu                   Purdue University
 *  A. Everett               Purdue University
 *  with contributions from: S. Lacaprara, J. Mumford, P. Traczyk
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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"


#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonSeedCleaner.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackReFitter.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"

#include "RecoMuon/TrackingTools/interface/MuonTrackConverter.h"
#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonMonitorInterface.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"

#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par,
							 const MuonServiceProxy* service, MuonTrackLoader* loader) : 
  theService(service), theTrackLoader(loader) {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|ctor";

  ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new MuonTrackReFitter(refitterPSet,theService);

  theLayerMeasurements = new MuonDetLayerMeasurements();

  theMakeTkSeedFlag = par.getParameter<bool>("RegionalSeedFlag");


  theTrackConverter = new MuonTrackConverter(par,theService);
  theTrackMatcher = new GlobalMuonTrackMatcher(par,theService);

  theMuonHitsOption = par.getParameter<int>("MuonHitsOption");
  theDirection = static_cast<NavigationDirection>(par.getParameter<int>("Direction"));
  thePtCut = par.getParameter<double>("PtCut");
  theProbCut = par.getParameter<double>("Chi2ProbabilityCut");
  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");

  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.0001,0.0,0.0001,0.0,0.0,28.09);

  theTkTrajsAvailableFlag = par.getParameter<bool>("TkTrajectoryAvailable");
  theFirstEvent = true;
  
  if(theMakeTkSeedFlag) {
    theL2SeededTkLabel = par.getParameter<std::string>("MuonSeededTracksInstance");
    theCkfBuilderName = par.getParameter<std::string>("TkTrackBuilder");
    ParameterSet seedGeneratorPSet = par.getParameter<ParameterSet>("SeedGeneratorParameters");
    theTkSeedGenerator = new TrackerSeedGenerator(seedGeneratorPSet,theService);
  } else {
    theTkTrackLabel = par.getParameter<edm::InputTag>("TkTrackCollectionLabel");
  }

  theMIMFlag = par.getUntrackedParameter<bool>("performMuonIntegrityMonitor",false);
  if(theMIMFlag) {
    LogInfo(category) << "Enabling Data Integrity Checks";
    dataMonitor = edm::Service<GlobalMuonMonitorInterface>().operator->(); 
  }

  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();

}


//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() {

  if (theRefitter) delete theRefitter;
  if (theTrackMatcher) delete theTrackMatcher;
  if (theLayerMeasurements) delete theLayerMeasurements;
  if (theTrackConverter) delete theTrackConverter;
  if (theTrajectoryCleaner) delete theTrajectoryCleaner;
}


//
// set Event
//
void GlobalMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|setEvent";

  theEvent = &event;

  // get tracker TrackCollection from Event
  edm::Handle<std::vector<Trajectory> > handleTrackerTrajs;
  if( !theMakeTkSeedFlag ) {
    event.getByLabel(theTkTrackLabel,allTrackerTracks);
    LogInfo(category) 
      << "Found " << allTrackerTracks->size() 
      << " tracker Tracks with label "<< theTkTrackLabel;  
    if ( theTkTrajsAvailableFlag ) {
      event.getByLabel(theTkTrackLabel,handleTrackerTrajs);
      allTrackerTrajs = &*handleTrackerTrajs;         
      if ( theFirstEvent ) LogInfo(category) << "Tk Trajectories Found! ";
    }
  }
  
  theLayerMeasurements->setEvent(event);  
  
  if( theMakeTkSeedFlag ) {   
    if (theFirstEvent) {
      theFirstEvent = false;
      LogInfo(category) << "Constructing a CkfBuilder";
      theService->eventSetup().get<CkfComponentsRecord>().get(theCkfBuilderName,theCkfBuilder);
    }
    theCkfBuilder->setEvent(event);
    theTkSeedGenerator->setEvent(event);
  }

}


//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::trajectories(const TrackCand& staCandIn) {


  if(theMIMFlag) {
    dataMonitor->book1D("cuts","events passing each cut",10,0.5,10.5);
    dataMonitor->fill1("cuts",1);
  }

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|trajectories";
  TimerStack timers;
  string timerName = category + "::Total";
  timers.push(timerName);

  // cut on muons with low momenta
  if ( (staCandIn).second->pt() < thePtCut || (staCandIn).second->innerMomentum().Rho() < thePtCut || (staCandIn).second->innerMomentum().R() < 2.5 ) return CandidateContainer();
  if(theMIMFlag) dataMonitor->fill1("cuts",2);

  // convert the STA track into a Trajectory if Trajectory not already present
  TrackCand staCand(staCandIn);
  addTraj(staCand);

  timerName = category + "::makeTkCandCollection";
  timers.push(timerName);
  vector<TrackCand> regionalTkTracks = makeTkCandCollection(staCand);
  LogInfo(category) << "Found " << regionalTkTracks.size() << " tracks within region of interest";  
  if(theMIMFlag && regionalTkTracks.size() > 0) dataMonitor->fill1("cuts",6);  
  // match tracker tracks to muon track
  timerName = category + "::trackMatcher";
  timers.pop_and_push(timerName);
  vector<TrackCand> trackerTracks = theTrackMatcher->match(staCand, regionalTkTracks);
  LogInfo(category) << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest";
  if(theMIMFlag && trackerTracks.size() > 0) dataMonitor->fill1("cuts",7);
  // build a combined tracker-muon MuonCandidate
  timerName = category + "::build";
  timers.pop_and_push(timerName);
  CandidateContainer result = build(staCand, trackerTracks);
  LogInfo(category) << "Found "<< result.size() << " GLBMuons from one STACand";

  if(theMIMFlag) {
    dataMonitor->book1D("glb_sta","GLB per STA",101,-0.5,100.5);
    dataMonitor->fill1("glb_sta",result.size());
    if(result.size() > 0) dataMonitor->fill1("cuts",8);
  }

  // free memory
  if ( staCandIn.first == 0) delete staCand.first;

  if ( !theTkTrajsAvailableFlag ) {
    for ( vector<TrackCand>::const_iterator is = regionalTkTracks.begin(); is != regionalTkTracks.end(); ++is) {
      delete (*is).first;   
    }
  }
  timers.clean_stack();
  return result;
  
}


//
// select tracks within the region of interest
//
vector<GlobalMuonTrajectoryBuilder::TrackCand> 
GlobalMuonTrajectoryBuilder::chooseRegionalTrackerTracks(const TrackCand& staCand, 
                                                         const vector<TrackCand>& tkTs) const {
  
  // define eta-phi region
  RectangularEtaPhiTrackingRegion regionOfInterest = defineRegionOfInterest(staCand.second);
  
  typedef PixelRecoRange< float > Range;
  typedef TkTrackingRegionsMargin< float > Margin;
  
  //Get region's etaRange and phiMargin
  Range etaRange = regionOfInterest.etaRange();
  Margin phiMargin = regionOfInterest.phiMargin();
  //Range phiRange(Geom::Phi<float>(regionOfInterest.direction().phi()) - fabs(Geom::Phi<float>(phiMargin.left())),Geom::Phi<float>(regionOfInterest.direction().phi()) + fabs(Geom::Phi<float>(phiMargin.right())));

  vector<TrackCand> result;

  vector<TrackCand>::const_iterator is;
  for ( is = tkTs.begin(); is != tkTs.end(); ++is ) {
    //check if each trackCand is in region of interest
    bool inEtaRange = etaRange.inside(is->second->eta());
    bool inPhiRange = (fabs(Geom::Phi<float>(is->second->phi()) - Geom::Phi<float>(regionOfInterest.direction().phi())) < phiMargin.right() ) ? true : false ;

    //for each trackCand in region, add trajectory and add to result
    if( inEtaRange && inPhiRange ) {
      TrackCand tmpCand = TrackCand(*is);
      addTraj(tmpCand);
      result.push_back(tmpCand);
    }
  }
  
  return result; 

}


//
// define a region of interest within the tracker
//
RectangularEtaPhiTrackingRegion GlobalMuonTrajectoryBuilder::defineRegionOfInterest(const reco::TrackRef& staTrack) const {

 //Get Track direction at vertex
  GlobalVector dirVector(staTrack->px(),staTrack->py(),staTrack->pz());

  //Get track momentum
  const math::XYZVector& mo = staTrack->innerMomentum();
  GlobalVector mom(mo.x(),mo.y(),mo.z());
  if ( staTrack->p() > 1.0 )
    mom = GlobalVector(staTrack->px(),staTrack->py(),staTrack->pz());

  //Get innerMu position
  const math::XYZPoint& po = staTrack->innerPosition();
  GlobalPoint pos(po.x(),po.y(),po.z());

  //Get dEta and dPhi: (direction at vertex) - (innerMuTsos position)
  float eta1 = dirVector.eta();
  float eta2 = pos.eta();
  float deta(fabs(eta1- eta2));
  float dphi(fabs(Geom::Phi<float>(dirVector.phi())-Geom::Phi<float>(pos.phi()))
);

  //deta = 1 * deta;
  //dphi = 1 * dphi;

  //deta = 1 * max(double(deta),0.05);
  //dphi = 1 * max(double(dphi),0.07);

  double minPt    = max(1.5,mom.perp()*0.6);
  double deltaZ   = min(15.9,3*sqrt(theVertexErr.czz()));
  double deltaEta = 0.05;
  double deltaPhi = 0.07;

  if ( deta > 0.05 ) { // 0.06
    deltaEta += deta/2;
  }
  if ( dphi > 0.07 ) {
    deltaPhi += 0.15;
    if ( fabs(eta2) < 1.0 && mom.perp() < 6. ) deltaPhi = dphi;
  }
  if ( fabs(eta1) < 1.25 && fabs(eta1) > 0.8 ) deltaEta = max(0.07,deltaEta);
  if ( fabs(eta1) < 1.3  && fabs(eta1) > 1.0 ) deltaPhi = max(0.3,deltaPhi);

  deltaEta = 1 * max(double(2.5 * deta),deltaEta);
  deltaPhi = 1 * max(double(3.5 * dphi),deltaPhi);

  RectangularEtaPhiTrackingRegion rectRegion(dirVector, theVertexPos,
                                             minPt, 0.2,
                                             deltaZ, deltaEta, deltaPhi);

  return rectRegion;

}


//
// build a combined tracker-muon trajectory
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::build(const TrackCand& staCand, 
                                                                     const std::vector<TrackCand>& tkMatchedTracks) const {

  // MuonHitsOption: 0 - tracker only
  //                 1 - include all muon hits
  //                 2 - include only first muon hit(s)
  //                 3 - include only selected muon hits
  //                 4 - combined
  //

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|build";

  if ( tkMatchedTracks.empty() ) return CandidateContainer();

  if (theMIMFlag) {
    dataMonitor->book1D("build","Passing each step of Build",11,-0.5,10.5);
    dataMonitor->fill1("build",1);
  }


  //
  // turn tkMatchedTracks into MuonCandidates
  //
  CandidateContainer tkTrajs;
  for (vector<TrackCand>::const_iterator tkt = tkMatchedTracks.begin(); tkt != tkMatchedTracks.end(); tkt++) {
    if ((*tkt).first != 0 && (*tkt).first->isValid()) {
      MuonCandidate* muonCand = new MuonCandidate(new Trajectory(*(*tkt).first),staCand.second,(*tkt).second);
      tkTrajs.push_back(muonCand);
    }
  }

  //
  // check and select muon measurements and 
  // measure occupancy in muon stations
  //   
  vector<int> stationHits(4,0);
  ConstRecHitContainer muonRecHits1; // all muon rechits
  ConstRecHitContainer muonRecHits2; // only first muon rechits
  if ( theMuonHitsOption > 0 ) checkMuonHits(*(staCand.second), muonRecHits1, muonRecHits2, stationHits);

  //
  // add muon hits and refit/smooth trajectories
  //
  CandidateContainer refittedResult;

  if ( theMuonHitsOption > 0 ) {

    for ( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); it++ ) {
      if(theMIMFlag) dataMonitor->fill1("build",2);
      // cut on tracks with low momenta
      const GlobalVector& mom = (*it)->trajectory()->lastMeasurement().updatedState().globalMomentum();
      if ( mom.mag() < 2.5 || mom.perp() < thePtCut ) continue;
      ConstRecHitContainer trackerRecHits = (*it)->trajectory()->recHits();
      if(theMIMFlag) dataMonitor->fill1("build",3);

      if ( theMakeTkSeedFlag && theDirection == insideOut ) {
	reverse(trackerRecHits.begin(),trackerRecHits.end());
	//sort(trackerRecHits.begin(),trackerRecHits.end(),RecHitLessByDet(alongMomentum));
      }

      if ( theDirection == insideOut ) {
	reverse(trackerRecHits.begin(),trackerRecHits.end());
      }
      
      TrajectoryMeasurement firstTM = ( theDirection == outsideIn || theMakeTkSeedFlag ) ? (*it)->trajectory()->firstMeasurement() : (*it)->trajectory()->lastMeasurement();
            
      TrajectoryStateOnSurface firstTsos = firstTM.updatedState();
      firstTsos.rescaleError(100.);
      
      //cout << "FirstTSOS Updated " <<endl 
      //     << firstTsos << endl;
      //<< firstTsos1.globalDirection() <<endl;
      
      if ( theMakeTkSeedFlag ) {
	TrajectoryMeasurement lastTM = ((*it)->trajectory()->direction() == alongMomentum) ? (*it)->trajectory()->lastMeasurement() : (*it)->trajectory()->firstMeasurement();
	TrajectoryStateOnSurface lastTsos = lastTM.updatedState();
	lastTsos.rescaleError(100.);
	
	//cout << "LastTSOS" << lastTsos << endl;
	//printHits(trackerRecHits);
	
	//TrajectoryStateTransform tsTransform;	
	TrajectoryStateOnSurface firstTsos2;
	if(trackerRecHits.front()->geographicalId().det() == DetId::Tracker ) {
	  firstTsos2 = theRefitter->propagator(oppositeToMomentum)->propagate(lastTsos,trackerRecHits.front()->det()->surface());
	}
	
	//if(firstTsos2.isValid()) cout << endl<< "Next TSOS Propagated " <<endl 
	//		      << firstTsos2 << endl;	
	firstTsos = firstTsos2;
      }      
      
      TC refitted1,refitted2,refitted3;
      vector<Trajectory*> refit(4);
      MuonCandidate* finalTrajectory = 0;

      // tracker only track
      refit[0] = (*it)->trajectory();
      ConstRecHitContainer rechits(trackerRecHits);

      // full track with all muon hits
      if ( theMuonHitsOption == 1 || theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
	rechits.insert(rechits.end(), muonRecHits1.begin(), muonRecHits1.end());
	if(theMIMFlag) dataMonitor->fill1("build",4);//should equal 3
	LogDebug(category) << "Number of hits: " << rechits.size();
	refitted1 = theRefitter->trajectories((*it)->trajectory()->seed(),rechits,firstTsos);

	if ( refitted1.size() == 1 ) {
	  if(theMIMFlag) dataMonitor->fill1("build",5);
	  refit[1] = &(*refitted1.begin());
	  if ( theMuonHitsOption == 1 ) {
            finalTrajectory = new MuonCandidate(new Trajectory(*refitted1.begin()), (*it)->muonTrack(), (*it)->trackerTrack());
             if ( (*it)->trajectory() ) delete (*it)->trajectory();
             if ( *it ) delete (*it);
          }
	}
      }

      // only first muon hits
      if ( theMuonHitsOption == 2 || theMuonHitsOption == 4 ) {
	rechits = trackerRecHits;
  	rechits.insert(rechits.end(), muonRecHits2.begin(), muonRecHits2.end());
	if(theMIMFlag) dataMonitor->fill1("build",6);
	LogDebug(category) << "Number of hits: " << rechits.size();
	
	refitted2 = theRefitter->trajectories((*it)->trajectory()->seed(),rechits,firstTsos);
	if ( refitted2.size() == 1 ) {
	  if(theMIMFlag) dataMonitor->fill1("build",7);
	  refit[2] = &(*refitted2.begin());
	  if ( theMuonHitsOption == 2 ) {
            finalTrajectory = new MuonCandidate(new Trajectory(*refitted2.begin()), (*it)->muonTrack(), (*it)->trackerTrack());
            if ( (*it)->trajectory() ) delete (*it)->trajectory();
            if ( *it ) delete (*it);
          }
	}
      } 

      // only selected muon hits
      if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
	ConstRecHitContainer muonRecHits3;
	if ( refitted1.size() == 1 ) muonRecHits3 = selectMuonHits(*refitted1.begin(),stationHits);
	rechits = trackerRecHits;
	rechits.insert(rechits.end(), muonRecHits3.begin(), muonRecHits3.end());
	
	LogDebug(category) << "Number of hits: " << rechits.size();
	if(theMIMFlag) dataMonitor->fill1("build",8);
	refitted3 = theRefitter->trajectories((*it)->trajectory()->seed(),rechits,firstTsos);
	if ( refitted3.size() == 1 ) {
	  if(theMIMFlag) dataMonitor->fill1("build",9);
	  refit[3] = &(*refitted3.begin());
	  if ( theMuonHitsOption == 3 ) {
            finalTrajectory = new MuonCandidate(new Trajectory(*refitted3.begin()), (*it)->muonTrack(), (*it)->trackerTrack());
            if ( (*it)->trajectory() ) delete (*it)->trajectory();
            if ( *it ) delete (*it);
          }
	}
      }

      if ( theMuonHitsOption == 4 ) {
	finalTrajectory = new MuonCandidate(new Trajectory(*chooseTrajectory(refit)), (*it)->muonTrack(), (*it)->trackerTrack());
        if ( (*it)->trajectory() ) delete (*it)->trajectory();
        if ( *it ) delete (*it);
      } 
      
      if ( finalTrajectory ) {
	if(theMIMFlag) dataMonitor->fill1("build",10);
	refittedResult.push_back(finalTrajectory);
      }
    }

  }
  else {
    refittedResult = tkTrajs;
  }
  //    int nRefitted = refittedResult.size();
  
  //FIXME: IMPLEMENT ME  
  //
  // muon trajectory cleaner
  //
  //    TrajectoryCleaner* mcleaner = new L3MuonTrajectoryCleaner();
  //    mcleaner->clean(refittedResult);
  //   delete mcleaner;
  
  //    if ( cout.testOut ) {
  //    cout.testOut << "seeds    : " << setw(3) << nSeeds << endl; 
  //      cout.testOut << "raw      : " << setw(3) << nRaw << endl;
  //    cout.testOut << "smoothed : " << setw(3) << nSmoothed << endl;
  //      cout.testOut << "cleaned  : " << setw(3) << nCleaned << endl;
  //      cout.testOut << "refitted : " << setw(3) << nRefitted << endl;
  //  }
  
  //
  //Perform a ghost suppression on all candidates, not only on those coming
  //from the same seed (RecMuon)
  //FIXME: IMPLEMENT ME

  return refittedResult;

}


//
//
//
void GlobalMuonTrajectoryBuilder::checkMuonHits(const reco::Track& muon, 
						ConstRecHitContainer& all,
						ConstRecHitContainer& first,
						std::vector<int>& hits) const {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|checkMuonHits";

  int dethits[4];
  for ( int i=0; i<4; i++ ) hits[i]=dethits[i]=0;
  
  ConstMuonRecHitContainer muonRecHits = theTrackConverter->getTransientMuonRecHits(muon);
  
  // loop through all muon hits and calculate the maximum # of hits in each chamber      
  for (ConstMuonRecHitContainer::const_iterator imrh = muonRecHits.begin(); imrh != muonRecHits.end(); imrh++ ) { 
    if ( !(*imrh)->isValid() ) continue;
    
    if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
      
      int station = 0;
      int detRecHits = 0;
      
      DetId id = (*imrh)->geographicalId();
      
      const DetLayer* layer = theService->detLayerGeometry()->idToLayer(id);
      MuonRecHitContainer dRecHits = theLayerMeasurements->recHits(layer);
      
      // get station of hit if it is in DT
      if ( (**imrh).isDT() ) {
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
                if ( all2dRecHits.size() > 0 ) {
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
	  LogDebug(category) << " Station " << station << " DT "<<(*ir)->dimension()<<" " << (*ir)->localPosition()
						      << " Distance: "<< rhitDistance<<" recHits: "<< detRecHits;
	}// end of for all2dRecHits
      }// end of if DT
      // get station of hit if it is in CSC
      else if ( (**imrh).isCSC() ) {
	CSCDetId did(id.rawId());
	station = did.station();
	
	float coneSize = 10.0;
	
	for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
	  double rhitDistance = ((**ir).localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
	  LogDebug(category) << " Station " << station << " CSC "<<(**ir).dimension()<<" "<<(**ir).localPosition()
                                                  << " Distance: "<< rhitDistance<<" recHits: "<<detRecHits;
	}
      }
      // get station of hit if it is in RPC
      else if ( (**imrh).isRPC() ) {
	RPCDetId rpcid(id.rawId());
	station = rpcid.station();
	float coneSize = 100.0;
	for (MuonRecHitContainer::const_iterator ir = dRecHits.begin(); ir != dRecHits.end(); ir++ ) {
	  double rhitDistance = ((**ir).localPosition()-(**imrh).localPosition()).mag();
	  if ( rhitDistance < coneSize ) detRecHits++;
	  LogDebug(category)<<" Station "<<station<<" RPC "<<(**ir).dimension()<<" "<< (**ir).localPosition()
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
    all.push_back((*imrh).get());
  } // end of loop over muon rechits
  if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 )  {
    for ( int i = 0; i < 4; i++ ) {
      LogDebug(category) <<"Station "<<i+1<<": "<<hits[i]<<" "<<dethits[i]; 
    }
  }     
  
  //
  // check order of muon measurements
  //
  if ((*all.begin())->globalPosition().mag() >
      (*(all.end()-1))->globalPosition().mag() ) {
    LogDebug(category)<< "reverse order: ";
    sort(all.begin(),all.end(),RecHitLessByDet(alongMomentum));
  }
  
  
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
	LogDebug(category) << "checkMuonHits:";
	LogDebug(category) << " station 1 = "<<station1 
						   <<", r = "<< (*ihit)->globalPosition().perp()
						   <<", z = "<< (*ihit)->globalPosition().z() << ", "; 
	
	LogDebug(category) << " station 2 = " << station2
						   <<", r = "<<(*(nexthit))->globalPosition().perp()
						   <<", z = "<<(*(nexthit))->globalPosition().z() << ", ";
	return;
      }
    }
    else if ( (nexthit == all.end()) && (station1 != -999) ) {
      LogDebug(category) << "checkMuonHits:";
      LogDebug(category) << " station 1 = "<< station1
                                              << ", r = " << (*ihit)->globalPosition().perp()
                                              << ", z = " << (*ihit)->globalPosition().z() << ", "; 
      return;
    }
  }
  // if none of the above is satisfied, return blank vector.
  first.clear();
  
}


//
// select muon hits compatible with trajectory; 
// check hits in chambers with showers
//
GlobalMuonTrajectoryBuilder::ConstRecHitContainer 
GlobalMuonTrajectoryBuilder::selectMuonHits(const Trajectory& traj, 
                                            const std::vector<int>& hits) const {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|selectMuonHits";
  ConstRecHitContainer muonRecHits;
  const double globalChi2Cut = 200.0;

  vector<TrajectoryMeasurement> muonMeasurements = traj.measurements(); 

  // loop through all muon hits and skip hits with bad chi2 in chambers with high occupancy      
  for (std::vector<TrajectoryMeasurement>::const_iterator im = muonMeasurements.begin(); im != muonMeasurements.end(); im++ ) {

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
      LogDebug(category)
	<< "Skip hit: " << id.det() << " " << station << ", " 
	<< chi2ndf << " (" << chi2Cut << " chi2 threshold) " 
	<< hits[station-1] << endl;
    }

  }
  
  //
  // check order of rechits
  //
  reverse(muonRecHits.begin(),muonRecHits.end());
  return muonRecHits;

}


//
// choose final trajectory
//
const Trajectory* GlobalMuonTrajectoryBuilder::chooseTrajectory(const std::vector<Trajectory*>& t) const {

  Trajectory* result = 0;
  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|chooseTrajectory";
 
  double prob0 = ( t[0] ) ? trackProbability(*t[0]) : 0.0;
  double prob1 = ( t[1] ) ? trackProbability(*t[1]) : 0.0;
  double prob2 = ( t[2] ) ? trackProbability(*t[2]) : 0.0;
  double prob3 = ( t[3] ) ? trackProbability(*t[3]) : 0.0; 

  LogDebug(category) << "Probabilities: " << prob0 << " " << prob1 << " " << prob2 << " " << prob3 << endl;

  if ( t[1] ) result = t[1];
  if ( (t[1] == 0) && t[3] ) result = t[3];
  
  if ( t[1] && t[3] && ( (prob1 - prob3) > 0.05 )  )  result = t[3];

  if ( t[0] && t[2] && fabs(prob2 - prob0) > theProbCut ) {
    LogDebug(category) << "select Tracker only: -log(prob) = " << prob0 << endl;
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

  return result;

}


//
// calculate the tail probability (-ln(P)) of a fit
//
double GlobalMuonTrajectoryBuilder::trackProbability(const Trajectory& track) const {

  int nDOF = 0;
  ConstRecHitContainer rechits = track.recHits();
  for ( ConstRecHitContainer::const_iterator i = rechits.begin(); i != rechits.end(); ++i ) {
    if ((*i)->isValid()) nDOF += (*i)->dimension();
  }
  
  nDOF = max(nDOF - 5, 0);
  double prob = -LnChiSquaredProbability(track.chiSquared(), nDOF);
  
  return prob;

}


//
// print RecHits
//
void GlobalMuonTrajectoryBuilder::printHits(const ConstRecHitContainer& hits) const {
  
  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|printHits";

  LogInfo(category) << "Used RecHits: " << hits.size();
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogInfo(category) << "invalid RecHit";
      continue; 
    }
    
    const GlobalPoint& pos = (*ir)->globalPosition();
    
    LogInfo(category) 
      << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y())
      << "  z = " << pos.z()
      << "  dimension = " << (*ir)->dimension()
      << "  " << (*ir)->det()->geographicalId().det()
      << "  " << (*ir)->det()->subDetector();
    /*
    cout 
      << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y())
      << "  z = " << pos.z()
      << "  dimension = " << (*ir)->dimension()
      << "  " << (*ir)->det()->geographicalId().det()
      << "  " << (*ir)->det()->subDetector() << endl;
    */
  }

}


//
// build a tracker Trajectory from a seed
//
GlobalMuonTrajectoryBuilder::TC GlobalMuonTrajectoryBuilder::makeTrajsFromSeeds(const vector<TrajectorySeed>& tkSeeds) const {

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|makeTrajsFromSeeds";
  TC result;
  
  LogInfo(category) << "Tracker Seeds from L2/STA Muon: " << tkSeeds.size();
  
  int nseed = 0;
  vector<Trajectory> rawResult;
  std::vector<TrajectorySeed>::const_iterator seed;
  for (seed = tkSeeds.begin(); seed != tkSeeds.end(); ++seed) {
    nseed++;
    LogDebug(category) << "Building a trajectory from seed " << nseed;
    
    TC tkTrajs = theCkfBuilder->trajectories(*seed);
    LogDebug(category) << "Trajectories from Seed " << tkTrajs.size();
    
    theTrajectoryCleaner->clean(tkTrajs);
    
    for(vector<Trajectory>::const_iterator it=tkTrajs.begin();
	it!=tkTrajs.end(); it++){
      if( it->isValid() ) {
	rawResult.push_back(*it);
      }
    }
    LogDebug(category) << "Trajectories from Seed after cleaning " << rawResult.size();
    
    //result.insert(result.end(), tkTrajs.begin(), tkTrajs.end());
    if(theMIMFlag) {
      dataMonitor->book1D("tk_seed","Tracks per seed",101,-0.5,100.5);
      dataMonitor->fill1("tk_seed",rawResult.size());
    }
  }
  //vector<Trajectory> unsmoothedResult;
  theTrajectoryCleaner->clean(rawResult);
  
  for (vector<Trajectory>::const_iterator itraw = rawResult.begin();
       itraw != rawResult.end(); itraw++) {
    if((*itraw).isValid()) result.push_back( *itraw);
  }
  
  LogInfo(category) << "Trajectories from all seeds " << result.size();
  return result;

}


//
// make a TrackCand collection using tracker Track, Trajectory information
//
vector<GlobalMuonTrajectoryBuilder::TrackCand> GlobalMuonTrajectoryBuilder::makeTkCandCollection(const TrackCand& staCand) const {
  


  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|makeTkCandCollection";
  TimerStack times;
  string timerName = category;
  times.push(timerName);

  vector<TrackCand> tkCandColl;
  
  // Tracks not available, make seeds and trajectories
  if ( theMakeTkSeedFlag ) {

    timerName = category + "::muonSeededTracking";
    times.push(timerName);

    LogDebug(category) << "Making Seeds";

    std::vector<TrajectorySeed> tkSeeds; 
    TC allTkTrajs;
    if( theMakeTkSeedFlag && staCand.first != 0  && staCand.first->isValid() ) {
      timerName = category + "::makeSeeds";
      times.push(timerName);
      RectangularEtaPhiTrackingRegion region = defineRegionOfInterest((staCand.second));
      tkSeeds = theTkSeedGenerator->trackerSeeds(*(staCand.first),region);

      LogDebug(category) << "Found " << tkSeeds.size() << " tracker seeds";

      if(theMIMFlag) {
	if(tkSeeds.size() > 0) dataMonitor->fill1("cuts",3);
	dataMonitor->book1D("seed_sta","Seeds per STA",101,-0.5,100.5);
	dataMonitor->fill1("seed_sta",tkSeeds.size());
      }
      timerName = category + "::makeTrajsFromSeed";
      times.pop_and_push(timerName);

      allTkTrajs = makeTrajsFromSeeds(tkSeeds);
      times.pop();

      MuonCandidate::TrajectoryContainer tmpTrajectoryContainer;
      for(TC::const_iterator iter = allTkTrajs.begin(); iter != allTkTrajs.end(); ++iter) {
	tmpTrajectoryContainer.push_back(new Trajectory(*iter));
      }
      //reco::TrackRefProd trackCollectionRefProd = theEvent->getRefBeforePut<reco::TrackCollection>(theL2SeededTkLabel);
      //theTrackLoader->loadTracks(tmpTrajectoryContainer,*theEvent,theL2SeededTkLabel);
      
      int position = 0;
      for (TC::const_iterator tt=allTkTrajs.begin();tt!=allTkTrajs.end();++tt){
	tkCandColl.push_back(TrackCand(new Trajectory(*tt),reco::TrackRef()));
	position++;
      } 
    }

    times.pop();
    LogDebug(category) << "Found " << tkCandColl.size() << " tkCands from seeds";

    if(theMIMFlag) {
      dataMonitor->book1D("tk_sta","Tracks per STA",101,-0.5,100.5);
      dataMonitor->fill1("tk_sta",tkCandColl.size());
    }

  } // Tracks are already in edm
  else {
    timerName = category + "::trackCollection";
    times.push(timerName);
    vector<TrackCand> tkTrackCands;
    for ( unsigned int position = 0; position != allTrackerTracks->size(); ++position ) {
      reco::TrackRef tkTrackRef(allTrackerTracks,position);
      TrackCand tkCand = TrackCand(0,tkTrackRef);
      if ( theTkTrajsAvailableFlag ) {
	timerName = category + "::addTrajectory";
	times.push(timerName);
	std::vector<Trajectory>::const_iterator it = allTrackerTrajs->begin()+position;
	const Trajectory* trajRef(&*it);
	if( trajRef->isValid() ) tkCand.first = trajRef;
	times.pop();
      } 
      tkTrackCands.push_back(tkCand);          
    }
    timerName = category + "::chooseRegionalTrackerTracks";
    times.push(timerName);
    if(theMIMFlag && tkTrackCands.size() > 0 ) dataMonitor->fill1("cuts",4);
    tkCandColl = chooseRegionalTrackerTracks(staCand,tkTrackCands);
  }
  times.clean_stack();
  if(theMIMFlag && tkCandColl.size() > 0) dataMonitor->fill1("cuts",5);
  return tkCandColl;

}


//
// add Trajectory* to TrackCand if not already present
//
void GlobalMuonTrajectoryBuilder::addTraj(TrackCand& candIn) const {

  TimerStack times;
  string timerName;

  const std::string category = "Muon|RecoMuon|GlobalMuonTrajectoryBuilder|addTraj";
  timerName = category;
  times.push(timerName);
  if( candIn.first == 0 ) {
    timerName = category + "::trackConvert";
    times.push(timerName);

    LogDebug(category) << "Making new trajectory from TrackRef " << (*candIn.second).pt();

    TC staTrajs = theTrackConverter->convert(candIn.second);
    candIn = ( !staTrajs.empty() ) ? TrackCand(new Trajectory(staTrajs.front()),candIn.second) : TrackCand(0,candIn.second);    

  }
  times.clean_stack(); 

}
