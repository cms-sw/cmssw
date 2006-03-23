#ifndef RecoMuon_TrackingTools_MuonTrajectoryBuilder_H
#define RecoMuon_TrackingTools_MuonTrajectoryBuilder_H

/** \class MuonTrajectoryBuilder
 *  Base class for the Muon reco Trajectory Builder 
 *
 *  $Date: 2006/03/21 13:29:48 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

namespace edm {class ParameterSet;}
class TrajectorySeed;


class MuonTrajectoryBuilder {
 public:
  typedef std::vector<Trajectory> TrajectoryContainer;
 public:
  
  /// Constructor with Parameter set
  MuonTrajectoryBuilder() {};
  MuonTrajectoryBuilder(const edm::ParameterSet& ) {};

  /// Destructor
  virtual ~MuonTrajectoryBuilder(){};

  /// Returns a vector of the reconstructed trajectories compatible with
  ///  the given seed.
      
  virtual TrajectoryContainer trajectories(const TrajectorySeed&) = 0;

 private:
  
 protected:

};
#endif
