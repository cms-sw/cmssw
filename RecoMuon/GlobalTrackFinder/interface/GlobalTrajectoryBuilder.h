#ifndef RecoMuon_TrackFinder_GlobalTrajectoryBuilder_H
#define RecoMuon_TrackFinder_GlobalTrajectoryBuilder_H

/** \class GlobalTrajectoryBuilder
 *  Concrete class for the GLB Muon reco 
 *
 *  $Date: 2006/03/23 15:15:36 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

// class TrajectorySeed; 
// class GlobalMuonRefitter; 
// class GlobalMuonBackwardFilter; 
// class GlobalMuonSmoother; 


namespace edm {class ParameterSet;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

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
