/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/05/18 12:29:27 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace edm;
  
StandAloneMuonTrajectoryBuilder::StandAloneMuonTrajectoryBuilder(const ParameterSet& par){

  // The max allowed eta (physical limit). Since it is the same both for the three filter, 
  // it has been placed here
  theMaxEta = par.getParameter<double>("EtaMaxAllowed");

  // The inward-outward fitter (starts from seed state)
  theRefitter = new StandAloneMuonRefitter(par);
  
  // The outward-inward fitter (starts from theRefitter outermost state)
  theBWFilter = new StandAloneMuonBackwardFilter(par);

  // The outward-inward fitter (starts from theBWFilter innermost state)
  theSmoother = new StandAloneMuonSmoother(par);
} 

void StandAloneMuonTrajectoryBuilder::setES(const edm::EventSetup& setup){
  // Get the Tracking Geometry
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  setup.get<IdealMagneticFieldRecord>().get(theMGField);
  
  // FIXME: move the above lines in the fitters!
  
  theRefitter->setES(setup);
  theBWFilter->setES(setup);
  theSmoother->setES(setup);
}

StandAloneMuonTrajectoryBuilder::~StandAloneMuonTrajectoryBuilder(){
  delete theRefitter;
  delete theBWFilter;
  delete theSmoother;
}


// FIXME, change trajL in another name

MuonTrajectoryBuilder::TrajectoryContainer 
StandAloneMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){ 

  std::string metname = "StandAloneMuonTrajectoryBuilder::trajectories";
  MuonPatternRecoDumper debug;

  // FIXME put a flag
  bool timing = false;
  TimeMe time_STABuilder_tot(metname,timing);

  TrajectoryContainer trajL;

  PTrajectoryStateOnDet pTSOD = seed.startingState();
  TrajectoryStateTransform tsTransform;
  const GeomDet* gdet = theTrackingGeometry->idToDet( DetId(pTSOD.detId()));
  

  TrajectoryStateOnSurface tsos = tsTransform.transientState(pTSOD, &(gdet->surface()), &*theMGField);
  FreeTrajectoryState ftk = *tsos.freeTrajectoryState();


  FreeTrajectoryState ftl(ftk);
  
  Trajectory theTraj(seed);
  //Trajectory theTraj(seed,oppositeToMomentum);
  
  LogDebug(metname)<< "---StandAloneMuonTrajectoryBuilder SEED:" << endl ;
  debug.dumpFTS(ftl,metname);
  
  
  if (fabs(ftl.momentum().eta())>theMaxEta) {
    LogDebug(metname) << "############################################################" << endl
		      << "StandAloneMuonTrajectoryBuilder: WARNING!! " << endl
		      << "The SeedGenerator delivers this Trajectory:" << endl;
    debug.dumpFTS(ftl,metname);
    LogDebug(metname) << "Such an high eta is unphysical and may lead to infinite loop" << endl
		      << "rejecting the Track." << endl
		      << "############################################################" << endl;
    return trajL;
  }

  // refine the FTS given by the seed
  refitter()->reset();
  
  static const string t1 = "StandAloneMuonTrajectoryBuilder::refitter";
  TimeMe timer1(t1,timing);
  refitter()->refit(ftl);
  
  int totalNofChamberUsed = refitter()->getTotalChamberUsed();
  // Get the outermost FTS
  ftl = refitter()->lastFTS();

  //@@SL 27-Jun-2002: sanity check for trajectory with very high eta, the true
  //problem is why we do reconstruct such problematics trajectories...
  if (fabs(ftl.momentum().eta())>theMaxEta) {
    LogDebug(metname) << "############################################################" << endl
		      << "StandAloneMuonTrajectoryBuilder: WARNING!! " << endl
		      << "At the end of TrajectoryRefitter the Trajectory is:" << endl;
    debug.dumpFTS(ftl,metname);
    LogDebug(metname) << "Such an high eta is unphysical and may lead to infinite loop" << endl
		      << "rejecting the Track." << endl
		      << "############################################################" << endl;
    return trajL;
  }
  
  LogDebug(metname) << "--- StandAloneMuonTrajectoryBuilder REFITTER OUTPUT " << endl ;
  debug.dumpFTS(ftl);
  LogDebug(metname) << "No of DT/CSC/RPC chamber used: " 
		    << refitter()->getDTChamberUsed()
		    << refitter()->getCSCChamberUsed() 
		    << refitter()->getRPCChamberUsed();
    
  

  return trajL;
}
