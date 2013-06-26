#ifndef RecoMuon_TrackingTools_MuonTrajectoryCleaner_H
#define RecoMuon_TrackingTools_MuonTrajectoryCleaner_H

/** \class MuonTrajectoryCleaner
 *  No description available.
 *
 *  $Date: 2010/03/24 19:52:41 $
 *  $Revision: 1.10 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>

//class Event;
class MuonTrajectoryCleaner {
 public:
  typedef MuonCandidate::TrajectoryContainer TrajectoryContainer;
  typedef MuonCandidate::CandidateContainer CandidateContainer;
  

  /// Constructor
  MuonTrajectoryCleaner() : reportGhosts_(false) {}

  /// Constructor for L2 muons (enable reportGhosts)
  MuonTrajectoryCleaner(bool reportGhosts) : reportGhosts_(reportGhosts) {}

  /// Destructor
  virtual ~MuonTrajectoryCleaner() {};

  // Operations

  /// Clean the trajectories container, erasing the (worst) clone trajectory
  void clean(TrajectoryContainer &muonTrajectories, edm::Event& evt); //used by reference...

  /// Clean the candidates container, erasing the (worst) clone trajectory
  void clean(CandidateContainer &muonTrajectories); //used by reference...

protected:

private:
  bool reportGhosts_;

};
#endif

