#ifndef GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: 2006/07/19 02:05:18 $
 *  $Revision: 1.11 $
 *  \author Chang Liu - Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonReconstructionEnumerators.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/EventSetup.h"
//#include "DataFormats/TrackReco/interface/TrackFwd.h"
//#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class RectangularEtaPhiTrackingRegion;
class TrajectoryStateOnSurface;
class MuonUpdatorAtVertex;
class MagneticField;
class GlobalMuonTrackMatcher;
class TransientTrackingRecHit;
//class TransientTrackBuilder;
class GenericTransientTrackingRecHitBuilder;
class GlobalTrackingGeometry;
class MuonDetLayerGeometry;
class GlobalMuonReFitter;
class Propagator;
class TrajectoryFitter;

namespace edm {class ParameterSet;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

public:

  typedef edm::OwnVector< const TransientTrackingRecHit>  RecHitContainer;
  typedef std::vector<Trajectory> TC;
  typedef TC::const_iterator TI;
  //typedef std::vector<Trajectory> TrajectoryContainer;
  //typedef std::pair<Trajectory, reco::TrackRef&> MuonCandidate;
  //typedef std::vector<MuonCandidate> CandidateContainer;
  //typedef TrajectoryContainer::const_iterator TCI;
 
  /** Constructor with Parameter Set */
  GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par);
          
  /** Destructor */
  ~GlobalMuonTrajectoryBuilder();

  /** Returns a vector of the reconstructed trajectories compatible with
   * the given seed.
   */
  CandidateContainer trajectories(const reco::TrackRef&);
  TrajectoryContainer trajectories(const TrajectorySeed&) {TrajectoryContainer result; return result;}
  
  std::vector<reco::TrackRef*> chosenTrackerTrackRef() const;

 private:
   
  // Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);
  
  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);

  //TrajectoryStateOnSurface muonToSurface(const reco::TrackRef&);

  std::vector<reco::TrackRef> chooseRegionalTrackerTracks(const reco::TrackRef&, const edm::Handle<reco::TrackCollection>& );

  RectangularEtaPhiTrackingRegion defineRegionOfInterest(const reco::TrackRef&);
  //  build combined trajectory from sta Track and tracker RecHits, 
  //  common for both options
  MuonTrajectoryBuilder::CandidateContainer build(const reco::TrackRef&, const std::vector<reco::TrackRef>);
  
  //  check muon RecHits, calculate chamber occupancy and select hits to be 
  //  used in the final fit
  void checkMuonHits(const reco::Track&, RecHitContainer&, RecHitContainer&, std::vector<int>&) const;
 
//  select muon hits compatible with trajectory; check hits in 
//  chambers with showers
    RecHitContainer selectMuonHits(const Trajectory&, const std::vector<int>&) const;
 
  // get TransientTrackingRecHits from Track
  RecHitContainer getTransientHits(const reco::Track&) const;

  // choose final trajectory
  const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&) const;

  // calculate chi2 probability (-ln(P))
  double trackProbability(const Trajectory&) const;    

  // get silicon tracker Trajectories from track Track and Seed directly
  TC getTkTrajFromTrack(const reco::TrackRef&) const;

  TC getTkTrajsFromTrack(const edm::ESHandle<TrajectoryFitter>&,
			 const edm::ESHandle<Propagator>&,
			 const RecHitContainer&,
			 TrajectoryStateOnSurface&,
			 const edm::Handle<TrajectorySeedCollection>&) const;
  
  // print all RecHits of a trajectory
  void printHits(const RecHitContainer&) const;

 private:

  edm::ParameterSet par_;
  
  GlobalPoint theVertexPos;
  GlobalError theVertexErr;
  MuonUpdatorAtVertex* theUpdator;
  GlobalMuonTrackMatcher* theTrackMatcher;
  GenericTransientTrackingRecHitBuilder* theGTTrackingRecHitBuilder;
  GlobalMuonReFitter* theRefitter;

  float theTrackMatcherChi2Cut;
  int   theMuonHitsOption;
  ReconstructionDirection theDirection;
  float thePtCut;
  float theProbCut;
  int   theHitThreshold;
  float theDTChi2Cut;
  float theCSCChi2Cut;
  float theRPCChi2Cut;
  
  std::string theFitterLabel;
  std::string thePropagatorLabel;
  std::string theSeedCollectionLabel;   
  //std::string theTransTrackBuilderLabel;   

  std::string theTkTrackLabel;

  edm::ESHandle<TrajectoryFitter> theFitter;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESHandle<MagneticField> theField;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
  //edm::ESHandle<TransientTrackBuilder> theTransTrackBuilder;

  edm::Handle<TrajectorySeedCollection> theSeeds; 
  edm::Handle<reco::TrackCollection> allTrackerTracks;

  //edm::ESHandle<TransientTrackingRecHitBuilder> theTransientHitBuilder;
  std::vector<reco::TrackRef*> theTkTrackRef; 

};
#endif
