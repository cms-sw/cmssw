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
 *  $Date: 2006/08/03 03:24:34 $
 *  $Revision: 1.26 $
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonSeedCleaner.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonReFitter.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/GlobalTrackFinder/interface/GlobalMuonTrackMatcher.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonTrajectoryBuilder::GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par) {
  
  ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new GlobalMuonReFitter(refitterPSet);
  
  ParameterSet updatorPSet = par.getParameter<ParameterSet>("UpdatorParameters");
  theUpdator = new MuonUpdatorAtVertex(updatorPSet);
  
  theTkTrackLabel = par.getParameter<string>("TkTrackCollectionLabel");
  theTTRHBuilderName = par.getParameter<string>("TTRHBuilder");

  theTrackMatcherChi2Cut = par.getParameter<double>("Chi2CutTrackMatcher");
  theMuonHitsOption = par.getParameter<int>("MuonHitsOption");
  theDirection = static_cast<ReconstructionDirection>(par.getParameter<int>("Direction"));
  thePtCut = par.getParameter<double>("PtCut");
  theProbCut = par.getParameter<double>("Chi2ProbabilityCut");
  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut  = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");

  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.0001,0.0,0.0001,0.0,0.0,28.09);

}


//--------------
// Destructor --
//--------------

GlobalMuonTrajectoryBuilder::~GlobalMuonTrajectoryBuilder() {

  if (theRefitter) delete theRefitter;
  if (theUpdator) delete theUpdator;
  if (theTrackMatcher) delete theTrackMatcher;

}


//
//
//
void GlobalMuonTrajectoryBuilder::setES(const edm::EventSetup& setup) {

  setup.get<IdealMagneticFieldRecord>().get(theField);  
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);
  
  setup.get<TransientRecHitRecord>().get(theTTRHBuilderName,theTkTransientTrackingRecHitBuilder);

  theUpdator->setES(setup);
  theTrackMatcher = new GlobalMuonTrackMatcher(theTrackMatcherChi2Cut,&*theField,&*theUpdator);
  theRefitter->setES(setup);

}


//
//
//
void GlobalMuonTrajectoryBuilder::setEvent(const edm::Event& event) {

  // get tracker TrackCollection from Event
  event.getByLabel(theTkTrackLabel,allTrackerTracks);
  LogInfo("GlobalMuonTrajectoryBuilder") 
  << "Found " << allTrackerTracks->size() << " tracker tracks with label "
  << theTkTrackLabel << endl;

}


//
//
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::trajectories(const reco::TrackRef& staTrack) {

  // select tracker tracks in eta-phi cone around muon
  vector<reco::TrackRef> regionalTkTracks = chooseRegionalTrackerTracks(staTrack,allTrackerTracks);
  LogInfo("GlobalMuonTrajectoryBuilder") << "Found " << regionalTkTracks.size() << " tracks within region of interest" << endl;

  // match tracker tracks to muon track
  vector<reco::TrackRef> trackerTracks = theTrackMatcher->match(staTrack, regionalTkTracks);
  LogInfo("GlobalMuonTrajectoryBuilder") << "Found " << trackerTracks.size() << " matching tracker tracks within region of interest" << endl;

  CandidateContainer result = build(staTrack, trackerTracks);
  
  return result;

}


//
// select tracks within the region of interest
//
vector<reco::TrackRef> 
GlobalMuonTrajectoryBuilder::chooseRegionalTrackerTracks(const reco::TrackRef& staTrack, 
                                                         const edm::Handle<reco::TrackCollection>& tkTs) const {

  // define eta-phi region
  RectangularEtaPhiTrackingRegion regionOfInterest = defineRegionOfInterest(staTrack);
  vector<reco::TrackRef> result;
  int position = 0;

  reco::TrackCollection::const_iterator is;
  for ( is = tkTs->begin(); is != tkTs->end(); ++is ) {

    position++;

    double deltaEta = 0.05;
    double deltaPhi = 0.07;

    float eta = is->innerMomentum().eta();
    float phi = is->innerMomentum().phi();

    float deta(fabs(eta-regionOfInterest.direction().eta()));
    float dphi(fabs(Geom::Phi<float>(phi)-Geom::Phi<float>(regionOfInterest.direction().phi())));
    if ( deta > deltaEta || dphi > deltaPhi ) continue;  
    //FIXME: should we limit the number of tracks in an area?
    reco::TrackRef tkTsRef(tkTs,position-1);
    result.push_back(tkTsRef); 
  }

  return result;

}


//
// define a region of interest within the tracker
//
RectangularEtaPhiTrackingRegion GlobalMuonTrajectoryBuilder::defineRegionOfInterest(const reco::TrackRef& staTrack) const
{

  // define tracker region of interest
  const math::XYZVector& mo = staTrack->innerMomentum();
  GlobalVector mom(mo.x(),mo.y(),mo.z());

  reco::TransientTrack staTT(staTrack,&*theField);
  TrajectoryStateOnSurface traj_vertex = staTT.impactPointState();
  if ( traj_vertex.isValid() ) mom = traj_vertex.globalMomentum();
  float eta1   = mom.eta();
  float phi1   = mom.phi();
  float theta1 = mom.theta();

  float eta2 = 0.0;
  float phi2 = 0.0;
  float theta2 = 0.0;
  
  ConstMuonRecHitContainer recHits = getTransientHits(*staTrack);
  ConstMuonRecHitPointer r = *(recHits.begin()+1);

  eta2   = r->globalPosition().eta();
  phi2   = r->globalPosition().phi();
  theta2 = r->globalPosition().theta();

  float deta(fabs(eta1-eta2));
  float dphi(fabs(Geom::Phi<float>(phi1)-Geom::Phi<float>(phi2)));

  double deltaEta = 0.05;
  double deltaPhi = 0.07; // 5*ephi;
  double deltaZ   = min(15.9,3*sqrt(theVertexErr.czz()));
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
  RectangularEtaPhiTrackingRegion rectRegion(direction, theVertexPos,
                                             minPt, 0.2, deltaZ, deltaEta, deltaPhi);

  return rectRegion;

}


//
//
//
MuonCandidate::CandidateContainer GlobalMuonTrajectoryBuilder::build(const reco::TrackRef& staTrack, 
                                                                     const std::vector<reco::TrackRef>& tkMatchedTracks) const {


  // MuonHitsOption: 0 - tracker only
  //                 1 - include all muon hits
  //                 2 - include only first muon hit(s)
  //                 3 - include only selected muon hits
  //                 4 - combined
  //

  //
  // turn tkMatchedTracks into tkTrajs
  //
  CandidateContainer tkTrajs;
 
  for (vector<reco::TrackRef>::const_iterator tkt = tkMatchedTracks.begin();tkt != tkMatchedTracks.end(); tkt++) {
    TC tkTrajs_tmp = getTrajFromTrack(*tkt);
    if ( !tkTrajs_tmp.empty() ) {
      Trajectory* traj = new Trajectory(tkTrajs_tmp.front());
      MuonCandidate* muonCand = new MuonCandidate(traj,staTrack,*tkt);
      tkTrajs.push_back(muonCand);
    }    
  }
  
  //
  // check and select muon measurements and 
  // measure occupancy of muon stations
  //   
  vector<int> stationHits(4,0);
  ConstRecHitContainer muonRecHits1; // all muon rechits
  ConstRecHitContainer muonRecHits2; // only first muon rechits
  if ( theMuonHitsOption > 0 ) checkMuonHits(*staTrack, muonRecHits1, muonRecHits2, stationHits);

  //
  // add muon hits and refit/smooth trajectories
  //
  CandidateContainer refittedResult;
  
  if ( theMuonHitsOption > 0 ) {
    
    for ( CandidateContainer::const_iterator it = tkTrajs.begin(); it != tkTrajs.end(); it++ ) {
    
      // cut on tracks with low momenta
      const GlobalVector& mom = (*it)->trajectory()->lastMeasurement().updatedState().globalMomentum();
      if ( mom.mag() < 2.5 || mom.perp() < thePtCut ) continue;
      ConstRecHitContainer trackerRecHits = (*it)->trajectory()->recHits();

      if ( theDirection == insideOut ) {
	reverse(trackerRecHits.begin(),trackerRecHits.end());
      }
      
      TrajectoryMeasurement firstTM = ( theDirection == outsideIn ) ? (*it)->trajectory()->firstMeasurement() : (*it)->trajectory()->lastMeasurement();
      TrajectoryStateOnSurface firstTsos = firstTM.updatedState();
      firstTsos.rescaleError(100.);
      
      TC refitted1,refitted2,refitted3;
      vector<Trajectory*> refit(4);
      MuonCandidate* finalTrajectory = 0;
      // tracker only track
      refit[0] = (*it)->trajectory();                 
      
      ConstRecHitContainer rechits(trackerRecHits);
      
      // full track with all muon hits
      if ( theMuonHitsOption == 1 || theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
	rechits.insert(rechits.end(), muonRecHits1.begin(), muonRecHits1.end());
	LogInfo("GlobalMuonTrajectoryBuilder") << "Number of hits: " << rechits.size();
        printHits(rechits);
	refitted1 = theRefitter->trajectories((*it)->trajectory()->seed(),rechits,firstTsos);
	if ( refitted1.size() == 1 ) {
	  refit[1] = &(*refitted1.begin());
	  if ( theMuonHitsOption == 1 ) finalTrajectory = new MuonCandidate(new Trajectory(*refitted1.begin()), (*it)->muonTrack(), (*it)->trackerTrack());
	}
      }
      
      // only first muon hits
      if ( theMuonHitsOption == 2 || theMuonHitsOption == 4 ) {
	rechits = trackerRecHits;
  	rechits.insert(rechits.end(), muonRecHits2.begin(), muonRecHits2.end());

	LogInfo("GlobalMuonTrajectoryBuilder")<< "Number of hits: "<<rechits.size();
	
	refitted2 = theRefitter->trajectories((*it)->trajectory()->seed(),rechits,firstTsos);
	if ( refitted2.size() == 1 ) {
	  refit[2] = &(*refitted2.begin());
	  if ( theMuonHitsOption == 2 ) finalTrajectory = new MuonCandidate(new Trajectory(*refitted2.begin()), (*it)->muonTrack(), (*it)->trackerTrack());
	}
      } 
      
      // only selected muon hits
      if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
	ConstRecHitContainer muonRecHits3;
	if ( refitted1.size() == 1 ) muonRecHits3 = selectMuonHits(*refitted1.begin(),stationHits);
	rechits = trackerRecHits;
	rechits.insert(rechits.end(), muonRecHits3.begin(), muonRecHits3.end());
	
	edm::LogInfo("GlobalMuonTrajectoryBuilder") << "Number of hits: " << rechits.size();
	
	refitted3 = theRefitter->trajectories((*it)->trajectory()->seed(),rechits,firstTsos);
	if ( refitted3.size() == 1 ) {
	  refit[3] = &(*refitted3.begin());
	  if ( theMuonHitsOption == 3 )  finalTrajectory = new MuonCandidate(new Trajectory(*refitted3.begin()), (*it)->muonTrack(), (*it)->trackerTrack());
	}
      }
      
      if ( theMuonHitsOption == 4 ) {
	finalTrajectory = new MuonCandidate(new Trajectory(*chooseTrajectory(refit)), (*it)->muonTrack(), (*it)->trackerTrack());
	
      } 
      
      if ( finalTrajectory ) {
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

  int dethits[4];
  for ( int i=0; i<4; i++ ) hits[i]=dethits[i]=0;
  
  MuonDetLayerMeasurements theLayerMeasurements;
  
  ConstMuonRecHitContainer muonRecHits = getTransientHits(muon);

  // loop through all muon hits and calculate the maximum # of hits in each chamber      
  for (ConstMuonRecHitContainer::const_iterator imrh = muonRecHits.begin(); imrh != muonRecHits.end(); imrh++ ) { 
    if ( !(*imrh)->isValid() ) continue;
    
    if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 ) {
      
      int station = 0;
      int detRecHits = 0;

      DetId id = (*imrh)->geographicalId();
      
      const DetLayer* layer = theDetLayerGeometry->idToLayer(id);
      MuonRecHitContainer dRecHits = theLayerMeasurements.recHits(layer);
      
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
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<<" Station "<<station<<" DT "<<(*ir)->dimension()<<" " << (*ir)->localPosition()
              <<" Distance: "<< rhitDistance<<" recHits: "<< detRecHits;
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
        edm::LogInfo("GlobalMuonTrajectoryBuilder")<<" Station "<<station<< " CSC "<<(**ir).dimension()<<" "<<(**ir).localPosition()
                  <<" Distance: "<< rhitDistance<<" recHits: "<<detRecHits;
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
    all.push_back((*imrh).get());
  } // end of loop over muon rechits
  if ( theMuonHitsOption == 3 || theMuonHitsOption == 4 )  {
    for ( int i = 0; i < 4; i++ ) {
      edm::LogInfo("GlobalMuonTrajectoryBuilder")<<"Station "<<i+1<<": "<<hits[i]<<" "<<dethits[i]; 
    }
  }     

  //
  // check order of muon measurements
  //
  if ((*all.begin())->globalPosition().mag() >
       (*(all.end()-1))->globalPosition().mag() ) {
     edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "reverse order: ";
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
          edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "checkMuonHits:";
          edm::LogInfo("GlobalMuonTrajectoryBuilder")<< " station 1 = "<<station1 
                        <<", r = "<< (*ihit)->globalPosition().perp()
              	        <<", z = "<< (*ihit)->globalPosition().z() << ", "; 
		        
	  edm::LogInfo("GlobalMuonTrajectoryBuilder")<< " station 2 = " << station2
						     <<", r = "<<(*(nexthit))->globalPosition().perp()
						     <<", z = "<<(*(nexthit))->globalPosition().z() << ", ";
	  return;
      }
    }
    else if ( (nexthit == all.end()) && (station1 != -999) ) {
      edm::LogInfo("GlobalMuonTrajectoryBuilder")<< "checkMuonHits:";
      edm::LogInfo("GlobalMuonTrajectoryBuilder")<< " station 1 = "<< station1
						 << ", r = " << (*ihit)->globalPosition().perp()
						 << ", z = " << (*ihit)->globalPosition().z() << ", "; 
      return;
    }
  }
  // if none of the above is satisfied, return blank vector.
  first.clear();

}


//
//
//
MuonTransientTrackingRecHit::ConstMuonRecHitContainer  
GlobalMuonTrajectoryBuilder::getTransientHits(const reco::Track& track) const {

   ConstMuonRecHitContainer result;
   for (trackingRecHit_iterator iter = track.recHitsBegin(); iter != track.recHitsEnd(); ++iter) {

     const TrackingRecHit* p = (*iter).get();
     const GeomDet* gd = theTrackingGeometry->idToDet(p->geographicalId());
   
     MuonRecHitPointer mp = MuonTransientTrackingRecHit::specificBuild(gd,p);    
     result.push_back(mp);

   }

   return result;

}


//
//
//
GlobalMuonTrajectoryBuilder::ConstRecHitContainer
GlobalMuonTrajectoryBuilder::getTkTransientHits(const reco::Track& track) const {

   ConstRecHitContainer result;

   for (trackingRecHit_iterator iter = track.recHitsBegin(); iter != track.recHitsEnd(); ++iter) {;
     result.push_back(theTkTransientTrackingRecHitBuilder->build(&**iter));
   }

   return result;

}


//
// select muon hits compatible with trajectory; 
// check hits in chambers with showers
//
GlobalMuonTrajectoryBuilder::ConstRecHitContainer 
GlobalMuonTrajectoryBuilder::selectMuonHits(const Trajectory& traj, 
                                            const std::vector<int>& hits) const {

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

    //LogDebug("GlobalMuonTrajectoryBuilder") << "hit: " << module << " " << station << " " << chi2ndf << " " << hits[station-1] << endl;

    bool keep = true;
    if ( (station > 0) && (station < 5) ) {
      if ( hits[station-1] > threshold ) keep = false;
    }   
    
    if ( (keep || ( chi2ndf < chi2Cut )) && ( chi2ndf < globalChi2Cut ) ) {
      muonRecHits.push_back((*im).recHit());
    } else {
      LogDebug("GlobalMuonTrajectoryBuilder")
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
 
  double prob0 = ( t[0] ) ? trackProbability(*t[0]) : 0.0;
  double prob1 = ( t[1] ) ? trackProbability(*t[1]) : 0.0;
  double prob2 = ( t[2] ) ? trackProbability(*t[2]) : 0.0;
  double prob3 = ( t[3] ) ? trackProbability(*t[3]) : 0.0; 

  LogDebug("GlobalMuonTrajectoryBuilder") << "Probabilities: " << prob0 << " " << prob1 << " " << prob2 << " " << prob3 << endl;

  if ( t[1] ) result = t[1];
  if ( (t[1] == 0) && t[3] ) result = t[3];
  
  if ( t[1] && t[3] && ( (prob1 - prob3) > 0.05 )  )  result = t[3];

  if ( t[0] && t[2] && fabs(prob2 - prob0) > theProbCut ) {
    LogDebug("GlobalMuonTrajectoryBuilder") << "select Tracker only: -log(prob) = " << prob0 << endl;
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
// convert a tracker Track into a Trajectory
//
GlobalMuonTrajectoryBuilder::TC GlobalMuonTrajectoryBuilder::getTrajFromTrack(const reco::TrackRef& tkTrack) const {

  TC result;

  // use TransientTrackingRecHitBuilder to get TransientTrackingRecHits 
  ConstRecHitContainer hits = getTkTransientHits((*tkTrack));

  // sort RecHits AlongMomentum
  reverse(hits.begin(),hits.end());
/*
  TransientTrackingRecHit::ConstRecHitPointer firstHit;
  for (ConstRecHitContainer::const_iterator it=tmp.begin(); it!=tmp.end();it++) {
    if ( (**it).isValid() ) {
      firstHit = *it;
      break;
    }
  }
  TransientTrackingRecHit::ConstRecHitPointer lastHit;
  for (ConstRecHitContainer::const_iterator it=tmp.end()-1; it!=tmp.begin()-1;it--) {
    if ( (**it).isValid() ) {
      lastHit = *it;
      break;
    }
  }
  if (firstHit->globalPosition().mag2() > (lastHit->globalPosition().mag2()) ) {
    for (ConstRecHitContainer::const_iterator it=tmp.end()-1;it!=tmp.begin()-1;it--){
      hits.push_back(*it);
    }
  } 
  else {
    hits = tmp;
  }
*/
  
  // use TransientTrackBuilder to get a starting TSOS
  reco::TransientTrack theTT(tkTrack,&*theField);
  TrajectoryStateOnSurface firstState = theRefitter->propagator()->propagate(theTT.impactPointState(), hits.front()->det()->surface());
  AlgebraicSymMatrix C(5,1);
  C *= 100.;
  TrajectoryStateOnSurface theTSOS(firstState.localParameters(), LocalTrajectoryError(C),
                                   firstState.surface(),
                                   &*theField);

  const TrajectorySeed* seed = new TrajectorySeed();
  vector<Trajectory> trajs = theRefitter->trajectories(*seed,hits,theTSOS);

  if ( !trajs.empty()) result.push_back(trajs.front()); 
  
  return result;

}


//
// print RecHits
//
void GlobalMuonTrajectoryBuilder::printHits(const ConstRecHitContainer& hits) const {

  LogInfo("GlobalMuonTrajectoryBuilder") << "Used RecHits: ";
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++ ) {
    if ( !(*ir)->isValid() ) {
      LogInfo("GlobalMuonTrajectoryBuilder") << "invalid RecHit";
      continue; 
    }

    const GlobalPoint& pos = (*ir)->globalPosition();
    LogInfo("GlobalMuonTrajectoryBuilder") 
    << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y())
    << "  z = " << pos.z()
    << "  dimension = " << (*ir)->dimension()
    << "  " << (*ir)->det()->geographicalId().det()
    << "  " << (*ir)->det()->subDetector();
  }

}
