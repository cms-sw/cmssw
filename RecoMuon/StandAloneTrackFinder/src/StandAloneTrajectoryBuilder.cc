/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/06/21 18:23:57 $
 *  $Revision: 1.15 $
 *  \author R. Bellan - INFN Torino
 *  \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
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
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/Navigation/interface/MuonNavigationSchool.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

// FIXME
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
//

using namespace edm;
using namespace std;

StandAloneMuonTrajectoryBuilder::StandAloneMuonTrajectoryBuilder(const ParameterSet& par){
  LogDebug("Muon|RecoMuon|StandAloneMuonTrajectoryBuilder") 
    << "constructor called" << endl;

  // The max allowed eta (physical limit). Since it is the same both for the three filter, 
  // it has been placed here
  theMaxEta = par.getParameter<double>("EtaMaxAllowed");

  // The inward-outward fitter (starts from seed state)
  ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new StandAloneMuonRefitter(refitterPSet);
  
  // The outward-inward fitter (starts from theRefitter outermost state)
  ParameterSet bwFilterPSet = par.getParameter<ParameterSet>("BWFilterParameters");
  //  theBWFilter = new StandAloneMuonBackwardFilter(bwFilterPSet); // FIXME
  theBWFilter = new StandAloneMuonRefitter(bwFilterPSet);


  // The outward-inward fitter (starts from theBWFilter innermost state)
  ParameterSet smootherPSet = par.getParameter<ParameterSet>("SmootherParameters");
  theSmoother = new StandAloneMuonSmoother(smootherPSet);
} 

void StandAloneMuonTrajectoryBuilder::setES(const EventSetup& setup){
  // Get the Tracking Geometry
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  setup.get<IdealMagneticFieldRecord>().get(theMGField);
  setup.get<MuonRecoGeometryRecord>().get(theDetLayerGeometry);
 
  // FIXME: move the above lines in the fitters!

  MuonNavigationSchool school(&*theDetLayerGeometry);
  NavigationSetter setter(school);
  
  theRefitter->setES(setup);
  theBWFilter->setES(setup);
  theSmoother->setES(setup);
}

void StandAloneMuonTrajectoryBuilder::setEvent(const edm::Event& event){
  theRefitter->setEvent(event);
  theBWFilter->setEvent(event);
  theSmoother->setEvent(event);
}

StandAloneMuonTrajectoryBuilder::~StandAloneMuonTrajectoryBuilder(){

  LogDebug("Muon|RecoMuon|StandAloneMuonTrajectoryBuilder") 
    << "StandAloneMuonTrajectoryBuilder destructor called" << endl;
  
  delete theRefitter;
  delete theBWFilter;
  delete theSmoother;
}


MuonTrajectoryBuilder::TrajectoryContainer 
StandAloneMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){ 

  std::string metname = "Muon|RecoMuon|StandAloneMuonTrajectoryBuilder";
  MuonPatternRecoDumper debug;

  // FIXME put a flag
  bool timing = false;
  TimeMe time_STABuilder_tot(metname,timing);

  // the trajectory container. In principle starting from one seed we can
  // obtain more than one trajectory. TODO: this feature is not yet implemented!
  TrajectoryContainer trajectoryContainer;
  
  // FIXME:check the prop direction!!
  Trajectory trajectoryFW(seed,alongMomentum);

  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();

  // Transform it in a TrajectoryStateOnSurface
  LogDebug(metname)<<"Transform PTrajectoryStateOnDet in a TrajectoryStateOnSurface"<<endl;
  TrajectoryStateTransform tsTransform;

  DetId seedDetId(pTSOD.detId());

  // FIXME!!! FIXME!!! FIXME!!! FIXME!!!
  const GeomDet* gdet = theTrackingGeometry->idToDet( seedDetId );
  TrajectoryStateOnSurface seedTSOS = tsTransform.transientState(pTSOD, &(gdet->surface()), &*theMGField);
  LogDebug(metname)<<"Seed Pt: "<<seedTSOS.freeState()->momentum().perp()<<endl;
  
  // FIXME!!! FIXME!!! FIXME!!! FIXME!!!
//   Surface::PositionType pos(0., 0., 0.);
//   Surface::RotationType rot;
//   BoundCylinder *cyl=
//     new BoundCylinder( pos, rot, SimpleCylinderBounds( 399., 401., -1200., 1200.));

//   TrajectoryStateOnSurface seedTSOS = tsTransform.transientState(pTSOD, cyl, &*theMGField);

  // FIXME remove this
  if(seedDetId.subdetId() == MuonSubdetId::CSC){
    CSCDetId cscId( seedDetId.rawId() );
    LogDebug(metname)<< "Seed id (CSC)"<< cscId << endl ;
  }
  else if (seedDetId.subdetId() == MuonSubdetId::DT){
       DTChamberId dtId( seedDetId.rawId() );
       LogDebug(metname)<< "Seed id (DT) "<< dtId << endl ;
  }
  //
  
  // Get the layer from which start the trajectory building
  const DetLayer *seedDetLayer = theDetLayerGeometry->idToLayer( seedDetId );

  LogDebug(metname)<< "---StandAloneMuonTrajectoryBuilder SEED:" << endl;
  debug.dumpTSOS(seedTSOS,metname);
  
  if (fabs(seedTSOS.globalMomentum().eta())>theMaxEta) {
    LogDebug(metname) << "############################################################" << endl
		      << "StandAloneMuonTrajectoryBuilder: WARNING!! " << endl
		      << "The SeedGenerator delivers this Trajectory:" << endl;
    debug.dumpTSOS(seedTSOS,metname);
    LogDebug(metname) << "Such an high eta is unphysical and may lead to infinite loop" << endl
		      << "rejecting the Track." << endl
		      << "############################################################" << endl;
    return trajectoryContainer;
  }

  // refine the FTS given by the seed
  static const string t1 = "StandAloneMuonTrajectoryBuilder::refitter";
  TimeMe timer1(t1,timing);

  // the trajectory is filled in the refitter::refit
  refitter()->refit(seedTSOS,seedDetLayer,trajectoryFW);

  // Get the last TSOS
  TrajectoryStateOnSurface tsosAfterRefit = refitter()->lastUpdatedTSOS();

  //@@SL 27-Jun-2002: sanity check for trajectory with very high eta, the true
  //problem is why we do reconstruct such problematics trajectories...
  if (fabs(tsosAfterRefit.globalMomentum().eta())>theMaxEta) {
    LogDebug(metname) << "############################################################" << endl
      		      << "StandAloneMuonTrajectoryBuilder: WARNING!! " << endl
		      << "At the end of TrajectoryRefitter the Trajectory is:" << endl;
    
    debug.dumpTSOS(tsosAfterRefit,metname);
    
    LogDebug(metname) << "Such an high eta is unphysical and may lead to infinite loop" << endl
		      << "rejecting the Track." << endl
		      << "############################################################" << endl;
    return trajectoryContainer;
  }

  LogDebug(metname) << "--- StandAloneMuonTrajectoryBuilder REFITTER OUTPUT " << endl ;
  debug.dumpTSOS(tsosAfterRefit,metname);
  

  if( refitter()->layers().size() ) debug.dumpLayer( refitter()->lastDetLayer(), metname);
  else return trajectoryContainer; // FIXME!!!!
  
  LogDebug(metname) << "Number of DT/CSC/RPC chamber used: " 
       << refitter()->getDTChamberUsed() << "/"
       << refitter()->getCSCChamberUsed() << "/"
       << refitter()->getRPCChamberUsed() <<endl;
  LogDebug(metname) << "Momentum: " <<tsosAfterRefit.freeState()->momentum();
  
  // FIXME put the possible choices: (factory???)
  // fw_low-granularity + bw_high-granularity
  // fw_high-granularity + smoother
  // fw_low-granularity + bw_high-granularity + smoother (not yet sure...)

  // BackwardFiltering
  // FIXME:check the prop direction!!
  Trajectory trajectoryBW(seed,oppositeToMomentum);

  static const string t2 = "StandAloneMuonTrajectoryBuilder::backwardfiltering";
  TimeMe timer2(t2,timing);

  bwfilter()->refit(tsosAfterRefit,refitter()->lastDetLayer(),trajectoryBW);

  // Get the last TSOS
  TrajectoryStateOnSurface tsosAfterBWRefit = bwfilter()->lastUpdatedTSOS();

  LogDebug(metname) << "--- StandAloneMuonTrajectoryBuilder BW FILTER OUTPUT " << endl ;
  debug.dumpTSOS(tsosAfterBWRefit,metname);
  LogDebug(metname) 
    << "Number of RecHits: " << trajectoryBW.foundHits() << endl
    << "Number of DT/CSC/RPC chamber used: " 
    << bwfilter()->getDTChamberUsed() << "/"
    << bwfilter()->getCSCChamberUsed() << "/" 
    << bwfilter()->getRPCChamberUsed();
  
  // The trajectory is good if there are at least 2 chamber used in total and at
  // least 1 "tracking" (DT or CSC)
  if (  bwfilter()->getTotalChamberUsed() >= 2 && 
	(bwfilter()->getDTChamberUsed() + bwfilter()->getCSCChamberUsed()) >0 ){
    LogDebug(metname)<< "TRAJECTORY SAVED" << endl;
    trajectoryContainer.push_back(trajectoryBW);
  }
  //if the trajectory is not saved, but at least two chamber are used in the
  //forward filtering, try to build a new trajectory starting from the old
  //trajectory w/o the latest measurement and a looser chi2 cut
  else if ( refitter()->getTotalChamberUsed() >= 2 ) {
    LogDebug(metname)<< "Trajectory NOT saved. SecondAttempt." << endl
		     << "FIRST MEASUREMENT KILLED" << endl; // FIXME: why???
    // FIXME:
    // a better choice could be: identify the poorest one, exclude it, redo
    // the fw and bw filtering. Or maybe redo only the bw without the excluded
    // measure. As first step I will port the ORCA algo, then I will move to the
    // above pattern.
    
    const string t2a="StandAloneMuonTrajectoryBuilder::backwardfilteringMuonTrackFinder:SecondAttempt";
    TimeMe timer2a(t2a,timing);

  }
  // smoother()->trajectories(trajectoryBW);
  return trajectoryContainer;
}
