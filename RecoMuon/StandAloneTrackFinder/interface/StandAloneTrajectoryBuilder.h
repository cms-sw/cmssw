#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/03/23 15:15:36 $
 *  $Revision: 1.2 $
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
  StandAloneMuonTrajectoryBuilder(const edm::ParameterSet& par);
          
  /** Destructor */
  virtual ~StandAloneMuonTrajectoryBuilder();

  /** Returns a vector of the reconstructed trajectories compatible with
   * the given seed.
   */   
  TrajectoryContainer trajectories(const TrajectorySeed&);

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
