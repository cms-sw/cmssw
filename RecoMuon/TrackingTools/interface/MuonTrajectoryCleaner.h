#ifndef RecoMuon_TrackingTools_MuonTrajectoryCleaner_H
#define RecoMuon_TrackingTools_MuonTrajectoryCleaner_H

/** \class MuonTrajectoryCleaner
 *  No description available.
 *
 *  $Date: 2006/04/26 06:54:15 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include <vector>

class MuonTrajectoryCleaner {
 public:
  typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
  typedef MuonCandidate::CandidateContainer CandidateContainer;
  

  /// Constructor
  MuonTrajectoryCleaner(){};

  /// Destructor
  virtual ~MuonTrajectoryCleaner(){};

  // Operations

  /// Clean the trajectories container, erasing the (worst) clone trajectory
  void clean(TrajectoryContainer &muonTrajectories); //used by reference...

protected:

private:

};
#endif

