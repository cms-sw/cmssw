#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/03/21 13:27:22 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

class TrajectorySeed;
class StandAloneMuonRefitter;
class StandAloneMuonBackwardFilter;
class StandAloneMuonSmoother;


namespace edm {class ParameterSet;}

class StandAloneMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

public:

  /** Constructor with Parameter set */
  StandAloneMuonTrajectoryBuilder(const edm::ParameterSet& par){} ;
          
  /** Destructor */
  ~StandAloneMuonTrajectoryBuilder(){};

  /** Returns a vector of the reconstructed trajectories compatible with
   * the given seed.
   */   
  TrajectoryContainer trajectories(const TrajectorySeed&){ return TrajectoryContainer();}

  StandAloneMuonRefitter* refitter() const {return theRefitter;}
  StandAloneMuonBackwardFilter* bwfilter() const {return theBWFilter;}
  StandAloneMuonSmoother* smoother() const {return theSmoother;}
  
 private:
  
  StandAloneMuonRefitter* theRefitter;
  StandAloneMuonBackwardFilter* theBWFilter;
  StandAloneMuonSmoother* theSmoother;


protected:

};
#endif
