#ifndef GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: $
 *  $Revision: $
 *  \author C. Liu - Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonReconstructionEnumerators.h"

class TrajectoryBuilder;
class TrajectorySmoother;
class TrajectoryCleaner;
class BasicTrajectorySeed; 
class GlobalMuonReFitter; 
class Muon;

namespace edm {class ParameterSet;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

public:

  typedef std::vector<Muon*> MuonContainer;
 
  /// constructor
  GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par){} ;
          
  /// destructor 
  ~GlobalMuonTrajectoryBuilder(){};

  /// reconstruct muon trajectories
  MuonContainer muons();

  private:

    struct TrajForRecMuon {
      TrajForRecMuon(const Trajectory& traj, MuonContainer::const_iterator muon, const Trajectory& tk) :
         theTrajectory(traj), theMuonRecMuon(muon), theTkTrajectory(tk) {
      }

      Trajectory theTrajectory;
      MuonContainer::const_iterator theMuonRecMuon;
      Trajectory theTkTrajectory;
    };

 private:
  
    MuonContainer* theMuons;

    TrajectoryBuilder*  theTrajectoryBuilder;
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
