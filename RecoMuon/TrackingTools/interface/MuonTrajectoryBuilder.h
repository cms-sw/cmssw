#ifndef RecoMuon_TrackingTools_MuonTrajectoryBuilder_H
#define RecoMuon_TrackingTools_MuonTrajectoryBuilder_H

/** \class MuonTrajectoryBuilder
 *  Base class for the Muon reco Trajectory Builder 
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino
 */

//FIXME??
#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"

namespace edm {class ParameterSet;}
class TrajectorySeed;

//FIXME 
class TrajectoryContainer{};

class MuonTrajectoryBuilder {
public:

  /// Constructor with Parameter set
  MuonTrajectoryBuilder() {};
  MuonTrajectoryBuilder(const edm::ParameterSet& ) {};

  /// Destructor
  virtual ~MuonTrajectoryBuilder(){};

  /// Returns a vector of the reconstructed trajectories compatible with
  ///  the given seed.
      
  virtual TrajectoryContainer trajectories(const TrackingSeed&) = 0;

 private:
  
 protected:

};
#endif
