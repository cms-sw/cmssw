/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  $Date: 2006/08/28 16:19:10 $
 *  $Revision: 1.25 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author Stefano Lacaprara - INFN Legnaro
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

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

StandAloneMuonTrajectoryBuilder::StandAloneMuonTrajectoryBuilder(const ParameterSet& par, 
								 const MuonServiceProxy* service):theService(service){
  LogDebug("Muon|RecoMuon|StandAloneMuonTrajectoryBuilder") 
    << "constructor called" << endl;

  // The inward-outward fitter (starts from seed state)
  ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
  theRefitter = new StandAloneMuonRefitter(refitterPSet,theService);

  // Disable/Enable the backward filter
  doBackwardRefit = par.getUntrackedParameter<bool>("DoBackwardRefit",true);
  
  if(doBackwardRefit){
    // The outward-inward fitter (starts from theRefitter outermost state)
    ParameterSet bwFilterPSet = par.getParameter<ParameterSet>("BWFilterParameters");
    //  theBWFilter = new StandAloneMuonBackwardFilter(bwFilterPSet,theService); // FIXME
    theBWFilter = new StandAloneMuonRefitter(bwFilterPSet,theService);

    theBWSeedType = bwFilterPSet.getParameter<string>("BWSeedType");
  }
  
  // The outward-inward fitter (starts from theBWFilter innermost state)
  ParameterSet smootherPSet = par.getParameter<ParameterSet>("SmootherParameters");
  theSmoother = new StandAloneMuonSmoother(smootherPSet,theService);
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
   if(doBackwardRefit) theBWFilter->setES(setup);
  theSmoother->setES(setup);
}

void StandAloneMuonTrajectoryBuilder::setEvent(const edm::Event& event){
  theRefitter->setEvent(event);
   if(doBackwardRefit) theBWFilter->setEvent(event);
  theSmoother->setEvent(event);
}

StandAloneMuonTrajectoryBuilder::~StandAloneMuonTrajectoryBuilder(){

  LogDebug("Muon|RecoMuon|StandAloneMuonTrajectoryBuilder") 
    << "StandAloneMuonTrajectoryBuilder destructor called" << endl;
  
  if(theRefitter) delete theRefitter;
  if(doBackwardRefit && theBWFilter) delete theBWFilter;
  if(theSmoother) delete theSmoother;
}


MuonTrajectoryBuilder::TrajectoryContainer 
StandAloneMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){ 

  const std::string metname = "Muon|RecoMuon|StandAloneMuonTrajectoryBuilder";
  MuonPatternRecoDumper debug;

  // FIXME put a flag
  bool timing = false;
  TimeMe time_STABuilder_tot(metname,timing);

  // the trajectory container. In principle starting from one seed we can
  // obtain more than one trajectory. TODO: this feature is not yet implemented!
  TrajectoryContainer trajectoryContainer;
  
  Trajectory trajectoryFW(seed,alongMomentum);

  // Get the Trajectory State on Det (persistent version of a TSOS) from the seed
  PTrajectoryStateOnDet pTSOD = seed.startingState();

  // Transform it in a TrajectoryStateOnSurface
  LogDebug(metname)<<"Transform PTrajectoryStateOnDet in a TrajectoryStateOnSurface"<<endl;
  TrajectoryStateTransform tsTransform;

  DetId seedDetId(pTSOD.detId());

  const GeomDet* gdet = theTrackingGeometry->idToDet( seedDetId );
  TrajectoryStateOnSurface seedTSOS = tsTransform.transientState(pTSOD, &(gdet->surface()), &*theMGField);
  LogDebug(metname)<<"Seed Pt: "<<seedTSOS.freeState()->momentum().perp()<<endl;

  LogDebug(metname)<< "Seed id in: "<< endl ;
  LogDebug(metname) << debug.dumpMuonId(seedDetId);
  
  // Get the layer from which start the trajectory building
  const DetLayer *seedDetLayer = theDetLayerGeometry->idToLayer( seedDetId );

  LogDebug(metname)<< "---StandAloneMuonTrajectoryBuilder SEED:" << endl;
  LogDebug(metname) << debug.dumpTSOS(seedTSOS);
  
  // refine the FTS given by the seed
  static const string t1 = "StandAloneMuonTrajectoryBuilder::refitter";
  TimeMe timer1(t1,timing);

  // the trajectory is filled in the refitter::refit
  refitter()->refit(seedTSOS,seedDetLayer,trajectoryFW);

  // Get the last TSOS
  TrajectoryStateOnSurface tsosAfterRefit = refitter()->lastUpdatedTSOS();

  LogDebug(metname) << "--- StandAloneMuonTrajectoryBuilder REFITTER OUTPUT " << endl ;
  LogDebug(metname) << debug.dumpTSOS(tsosAfterRefit);
  

  if( refitter()->layers().size() ) 
    LogDebug(metname) << debug.dumpLayer( refitter()->lastDetLayer());
  else return trajectoryContainer; 
  
  LogDebug(metname) << "Number of DT/CSC/RPC chamber used (fw): " 
       << refitter()->getDTChamberUsed() << "/"
       << refitter()->getCSCChamberUsed() << "/"
       << refitter()->getRPCChamberUsed() <<endl;
  LogDebug(metname) << "Momentum: " <<tsosAfterRefit.freeState()->momentum();
  

  if(!doBackwardRefit){
    LogDebug(metname) << "Only forward refit requested. Any backward refit will be performed!"<<endl;
    
    if (  refitter()->getTotalChamberUsed() >= 2 && 
	  (refitter()->getDTChamberUsed() + refitter()->getCSCChamberUsed()) >0 ){
      LogDebug(metname)<< "Trajectory saved" << endl;
      //FIXME! creating with new!
      trajectoryContainer.push_back(new Trajectory(trajectoryFW));
    }
    else LogDebug(metname)<< "Trajectory NOT saved. No enough number of tracking chamber used!" << endl;
    
    return trajectoryContainer;
  }

  // FIXME put the possible choices: (factory???)
  // fw_low-granularity + bw_high-granularity
  // fw_high-granularity + smoother
  // fw_low-granularity + bw_high-granularity + smoother (not yet sure...)

  // BackwardFiltering

  TrajectorySeed seedForBW;

  if(theBWSeedType == "noSeed"){
    TrajectorySeed seedVoid;
    seedForBW = seedVoid;
  }
  else if (theBWSeedType == "fromFWFit"){
    
    TrajectoryStateTransform tsTransform;
    
    PTrajectoryStateOnDet *seedTSOS =
      tsTransform.persistentState( tsosAfterRefit, trajectoryFW.lastMeasurement().recHit()->geographicalId().rawId());
    
    edm::OwnVector<TrackingRecHit> recHitsContainer; // FIXME!!
    TrajectorySeed fwFit(*seedTSOS,recHitsContainer,alongMomentum);

    seedForBW = fwFit;
  }
  else if (theBWSeedType == "fromGenerator"){
    seedForBW = seed;
  }
  else
    LogWarning(metname) << "Wrong seed type for the backward filter!";

  Trajectory trajectoryBW(seedForBW,oppositeToMomentum);

  static const string t2 = "StandAloneMuonTrajectoryBuilder::backwardfiltering";
  TimeMe timer2(t2,timing);

  // FIXME! under check!
  //  bwfilter()->refit(tsosAfterRefit,refitter()->lastDetLayer(),trajectoryBW);
  bwfilter()->refit(trajectoryFW.lastMeasurement().predictedState(),refitter()->lastDetLayer(),trajectoryBW);

  // Get the last TSOS
  TrajectoryStateOnSurface tsosAfterBWRefit = bwfilter()->lastUpdatedTSOS();

  LogDebug(metname) << "--- StandAloneMuonTrajectoryBuilder BW FILTER OUTPUT " << endl ;
  LogDebug(metname) << debug.dumpTSOS(tsosAfterBWRefit);
  LogDebug(metname) 
    << "Number of RecHits: " << trajectoryBW.foundHits() << "\n"
    << "Number of DT/CSC/RPC chamber used (bw): " 
    << bwfilter()->getDTChamberUsed() << "/"
    << bwfilter()->getCSCChamberUsed() << "/" 
    << bwfilter()->getRPCChamberUsed();
  
  // The trajectory is good if there are at least 2 chamber used in total and at
  // least 1 "tracking" (DT or CSC)
  if (  bwfilter()->getTotalChamberUsed() >= 2 && 
	(bwfilter()->getDTChamberUsed() + bwfilter()->getCSCChamberUsed()) >0 ){
    LogDebug(metname)<< "Trajectory saved" << endl;
     //FIXME! creating with new!
    trajectoryContainer.push_back(new Trajectory(trajectoryBW));
  }
  //if the trajectory is not saved, but at least two chamber are used in the
  //forward filtering, try to build a new trajectory starting from the old
  //trajectory w/o the latest measurement and a looser chi2 cut
  else if ( refitter()->getTotalChamberUsed() >= 2 ) {
    LogDebug(metname)<< "Trajectory NOT saved. Second Attempt." << endl
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
