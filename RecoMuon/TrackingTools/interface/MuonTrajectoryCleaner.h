#ifndef RecoMuon_TrackingTools_MuonTrajectoryCleaner_H
#define RecoMuon_TrackingTools_MuonTrajectoryCleaner_H

/** \class MuonTrajectoryCleaner
 *  No description available.
 *
 *  $Date: 2006/03/21 13:29:48 $
 *  $Revision: 1.1 $
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
  void clean(TrajectoryContainer &muonTrajectories){}; //used by reference...

protected:

private:

};
#endif

