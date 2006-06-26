#ifndef GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: 2006/06/18 19:14:51 $
 *  $Revision: 1.2 $
 *  \author C. Liu - Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonReconstructionEnumerators.h"

class CkfTrajectoryBuilder;
class TrajectorySmoother;
class TrajectoryCleaner;
class BasicTrajectorySeed; 
class GlobalMuonReFitter; 
class Muon;

namespace edm {class ParameterSet;}

class GlobalMuonTrajectoryBuilder{

public:

  typedef std::vector<Muon*> MuonCollection;
  typedef edm::OwnVector< const TransientTrackingRecHit>  RecHitContainer;
  typedef std::vector<Trajectory> TC;
  typedef TC::const_iterator TI;

 
  /// constructor
  GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par);
          
  /// destructor 
  ~GlobalMuonTrajectoryBuilder();

  /// reconstruct muon trajectories
  MuonCollection muons();

  private:

    struct TrajForRecMuon {
      TrajForRecMuon(const Trajectory& traj, MuonCollection::const_iterator muon, const Trajectory& tk) :
         theTrajectory(traj), theMuonRecMuon(muon), theTkTrajectory(tk) {
      }

      Trajectory theTrajectory;
      MuonCollection::const_iterator theMuonRecMuon;
      Trajectory theTkTrajectory;
    };

  private:

    /// initialize algorithms
    void init();
 
    /// get silicon tracker tracks
    std::vector<Trajectory> getTrackerTracks(const Muon&, int&, int&, int&, int&) const;

    /// check muon RecHits
    void checkMuonHits(const Muon&, RecHitContainer&, RecHitContainer&, std::vector<int>&) const;

    /// select muon RecHits
    RecHitContainer selectMuonHits(const Trajectory&, const std::vector<int>&) const;

    /// check candidates
    void checkMuonCandidates(MuonCollection&) const;

    /// choose final trajectory
    const Trajectory* chooseTrajectory(const std::vector<Trajectory*>&) const;

    /// match two trajectories
    TC::const_iterator matchTrajectories(const Trajectory&, TC&) const;

    /// calculate chi2 probability (-ln(P))
    double trackProbability(const Trajectory&) const;    

    /// print all RecHits of a trajectory
    void printHits(const RecHitContainer&) const;

    /// convert trajectories to RecMuons
    MuonCollection convertToRecMuons(std::vector<TrajForRecMuon>&) const;


 private:
  
    MuonCollection* theMuons;

    CkfTrajectoryBuilder*  theTrajectoryBuilder;
    TrajectorySmoother* theTrajectorySmoother;
    TrajectoryCleaner*  theTrajectoryCleaner;
    GlobalMuonReFitter* theRefitter;

    ReconstructionDirection theDirection;
    int   theMuonHitsOption;
    float thePtCut;
    float theProbCut;
    int   theHitThreshold;
    float theDTChi2Cut;
    float theCSCChi2Cut;
    float theRPCChi2Cut;

};
#endif
