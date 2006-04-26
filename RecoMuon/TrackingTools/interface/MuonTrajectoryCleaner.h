#ifndef RecoMuon_TrackingTools_MuonTrajectoryCleaner_H
#define RecoMuon_TrackingTools_MuonTrajectoryCleaner_H

/** \class MuonTrajectoryCleaner
 *  No description available.
 *
 *  $Date: 2006/03/23 15:15:37 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

class MuonTrajectoryCleaner {

 public:
  
  typedef std::vector<Trajectory> TrajectoryContainer;
  
 public:
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

