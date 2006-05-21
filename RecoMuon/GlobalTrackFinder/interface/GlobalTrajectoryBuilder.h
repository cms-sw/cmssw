#ifndef RecoMuon_TrackFinder_GlobalTrajectoryBuilder_H
#define RecoMuon_TrackFinder_GlobalTrajectoryBuilder_H

/** \class GlobalTrajectoryBuilder
 *  Concrete class for the GLB Muon reco 
 *
 *  $Date: 2006/05/17 13:05:13 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 *  \author C. Liu - Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

class TrajectoryBuilder;
class TrajectorySmoother;
class TrajectoryCleaner;
class BasicTrajectorySeed; 
class GlobalMuonReFitter; 

namespace edm {class ParameterSet;}

class GlobalTrajectoryBuilder : public MuonTrajectoryBuilder{

public:

  /** Constructor with Parameter set */
  GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par){} ;
          
  /** Destructor */
  ~GlobalMuonTrajectoryBuilder(){};

  /** Returns a vector of the reconstructed trajectories compatible with
   * the given seed.
   */   
  TrajectoryContainer trajectories(const TrajectorySeed&){ return TrajectoryContainer();}

  // GlobalMuonRefitter* refitter() const {return theRefitter;}
  // GlobalMuonBackwardFilter* bwfilter() const {return theBWFilter;}
  // GlobalMuonSmoother* smoother() const {return theSmoother;}
  
 private:
  
  // GlobalMuonRefitter* theRefitter;
  // GlobalMuonBackwardFilter* theBWFilter;
  // GlobalMuonSmoother* theSmoother;


protected:

};
#endif
