#ifndef RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H
#define RecoMuon_TrackFinder_StandAloneTrajectoryBuilder_H

/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

class StandAloneMuonFTSRefiner;
class StandAloneMuonBackwardFilter;
class StandAloneMuonSmoother;

//FIXME??
#include "DataFormats/TrackingSeed/interface/TrackingSeed.h"


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
  TrajectoryContainer trajectories(const TrackingSeed&){ return TrajectoryContainer();}

  StandAloneMuonFTSRefiner* refiner() const {return theRefiner;}
  StandAloneMuonBackwardFilter* bwfilter() const {return theBWFilter;}
  StandAloneMuonSmoother* smoother() const {return theSmoother;}
  
 private:
  
  StandAloneMuonFTSRefiner* theRefiner;
  StandAloneMuonBackwardFilter* theBWFilter;
  StandAloneMuonSmoother* theSmoother;


protected:

};
#endif
