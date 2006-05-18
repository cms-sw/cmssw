/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/05/18 09:53:21 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"

#include "Utilities/Timing/interface/TimingReport.h"

using namespace edm;
  
StandAloneMuonTrajectoryBuilder::StandAloneMuonTrajectoryBuilder(const ParameterSet& par){
  
  // The inward-outward fitter (starts from seed state)
  theRefitter = new StandAloneMuonRefitter(par);
  
  // The outward-inward fitter (starts from theRefitter outermost state)
  theBWFilter = new StandAloneMuonBackwardFilter(par);

  // The outward-inward fitter (starts from theBWFilter innermost state)
  theSmoother = new StandAloneMuonSmoother(par);
} 

StandAloneMuonTrajectoryBuilder::~StandAloneMuonTrajectoryBuilder(){
  delete theRefitter;
  delete theBWFilter;
  delete theSmoother;
}


MuonTrajectoryBuilder::TrajectoryContainer 
StandAloneMuonTrajectoryBuilder::trajectories(const TrajectorySeed&){ 

  // FIXME out a flag
  TimeMe time_STABuilder_tot("StandAloneMuonTrajectoryBuilder:total",false);

  TrajectoryContainer trajL;

  return TrajectoryContainer();
}
