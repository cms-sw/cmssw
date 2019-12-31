/** \class StandAloneTrajectoryBuilder
 *  Concrete class for the STA Muon reco 
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author Stefano Lacaprara - INFN Legnaro
 *  \author D. Trocino - INFN Torino <daniele.trocino@to.infn.it>
 *
 *  Modified by C. Calabria
 *  Modified by D. Nash
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonFilter.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TrackRefitter/interface/SeedTransformer.h"

using namespace edm;
using namespace std;

StandAloneMuonTrajectoryBuilder::StandAloneMuonTrajectoryBuilder(const ParameterSet& par,
                                                                 const MuonServiceProxy* service,
                                                                 edm::ConsumesCollector& iC)
    : theService(service) {
  const std::string metname = "Muon|RecoMuon|StandAloneMuonTrajectoryBuilder";

  LogTrace(metname) << "constructor called" << endl;

  // The navigation type:
  // "Direct","Standard"
  theNavigationType = par.getParameter<string>("NavigationType");

  // The inward-outward fitter (starts from seed state)
  ParameterSet filterPSet = par.getParameter<ParameterSet>("FilterParameters");
  filterPSet.addParameter<string>("NavigationType", theNavigationType);
  theFilter = new StandAloneMuonFilter(filterPSet, theService, iC);

  // Fit direction
  string seedPosition = par.getParameter<string>("SeedPosition");

  if (seedPosition == "in")
    theSeedPosition = recoMuon::in;
  else if (seedPosition == "out")
    theSeedPosition = recoMuon::out;
  else
    throw cms::Exception("StandAloneMuonFilter constructor")
        << "Wrong seed position chosen in StandAloneMuonFilter::StandAloneMuonFilter ParameterSet"
        << "\n"
        << "Possible choices are:"
        << "\n"
        << "SeedPosition = in or SeedPosition = out";

  // Propagator for the seed extrapolation
  theSeedPropagatorName = par.getParameter<string>("SeedPropagator");

  // Disable/Enable the backward filter
  doBackwardFilter = par.getParameter<bool>("DoBackwardFilter");

  // Disable/Enable the refit of the trajectory
  doRefit = par.getParameter<bool>("DoRefit");

  // Disable/Enable the refit of the trajectory seed
  doSeedRefit = par.getParameter<bool>("DoSeedRefit");

  if (doBackwardFilter) {
    // The outward-inward fitter (starts from theFilter outermost state)
    ParameterSet bwFilterPSet = par.getParameter<ParameterSet>("BWFilterParameters");
    //  theBWFilter = new StandAloneMuonBackwardFilter(bwFilterPSet,theService); // FIXME
    bwFilterPSet.addParameter<string>("NavigationType", theNavigationType);
    theBWFilter = new StandAloneMuonFilter(bwFilterPSet, theService, iC);

    theBWSeedType = bwFilterPSet.getParameter<string>("BWSeedType");
  }

  if (doRefit) {
    // The outward-inward fitter (starts from theBWFilter innermost state)
    ParameterSet refitterPSet = par.getParameter<ParameterSet>("RefitterParameters");
    theRefitter = new StandAloneMuonRefitter(refitterPSet, theService);
  }

  // The seed transformer (used to refit the seed and get the seed transient state)
  //  ParameterSet seedTransformerPSet = par.getParameter<ParameterSet>("SeedTransformerParameters");
  ParameterSet seedTransformerParameters = par.getParameter<ParameterSet>("SeedTransformerParameters");
  theSeedTransformer = new SeedTransformer(seedTransformerParameters);
}

void StandAloneMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  theFilter->setEvent(event);
  if (doBackwardFilter)
    theBWFilter->setEvent(event);
}

StandAloneMuonTrajectoryBuilder::~StandAloneMuonTrajectoryBuilder() {
  LogTrace("Muon|RecoMuon|StandAloneMuonTrajectoryBuilder")
      << "StandAloneMuonTrajectoryBuilder destructor called" << endl;

  if (theFilter)
    delete theFilter;
  if (doBackwardFilter && theBWFilter)
    delete theBWFilter;
  if (doRefit && theRefitter)
    delete theRefitter;
  if (theSeedTransformer)
    delete theSeedTransformer;
}

namespace {
  struct Resetter {
    StandAloneMuonFilter* mf;
    explicit Resetter(StandAloneMuonFilter* imf) : mf(imf) {}
    ~Resetter() {
      if (mf)
        mf->reset();
    }
  };
}  // namespace

MuonTrajectoryBuilder::TrajectoryContainer StandAloneMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed) {
  Resetter fwReset(filter());
  Resetter bwReset(bwfilter());

  const std::string metname = "Muon|RecoMuon|StandAloneMuonTrajectoryBuilder";
  MuonPatternRecoDumper debug;

  // Set the services for the seed transformer
  theSeedTransformer->setServices(theService->eventSetup());

  // the trajectory container. In principle starting from one seed we can
  // obtain more than one trajectory. TODO: this feature is not yet implemented!
  TrajectoryContainer trajectoryContainer;

  PropagationDirection fwDirection = (theSeedPosition == recoMuon::in) ? alongMomentum : oppositeToMomentum;
  Trajectory trajectoryFW(seed, fwDirection);

  TrajectoryStateOnSurface lastTSOS;
  DetId lastDetId;

  vector<Trajectory> seedTrajectories;

  if (doSeedRefit) {
    seedTrajectories = theSeedTransformer->seedTransform(seed);
    if (!seedTrajectories.empty()) {
      TrajectoryMeasurement lastTM(seedTrajectories.front().lastMeasurement());
      lastTSOS = lastTM.updatedState();
      lastDetId = lastTM.recHit()->geographicalId();
    }
  }

  if (!doSeedRefit || seedTrajectories.empty()) {
    lastTSOS = theSeedTransformer->seedTransientState(seed);
    lastDetId = seed.startingState().detId();
  }

  LogTrace(metname) << "Trajectory State on Surface before the extrapolation" << endl;
  LogTrace(metname) << debug.dumpTSOS(lastTSOS);

  // Segment layer
  LogTrace(metname) << "The RecSegment relies on: " << endl;
  LogTrace(metname) << debug.dumpMuonId(lastDetId);

  DetLayerWithState inputFromSeed = propagateTheSeedTSOS(lastTSOS, lastDetId);

  // refine the FTS given by the seed

  // the trajectory is filled in the refitter::refit
  filter()->refit(inputFromSeed.second, inputFromSeed.first, trajectoryFW);

  // "0th order" check...
  if (trajectoryFW.empty()) {
    LogTrace(metname) << "Forward trajectory EMPTY! No trajectory will be loaded!" << endl;
    return trajectoryContainer;
  }

  // Get the last TSOS
  //  TrajectoryStateOnSurface tsosAfterRefit = filter()->lastUpdatedTSOS();     // this is the last UPDATED TSOS
  TrajectoryStateOnSurface tsosAfterRefit = filter()->lastCompatibleTSOS();  // this is the last COMPATIBLE TSOS

  LogTrace(metname) << "StandAloneMuonTrajectoryBuilder filter output " << endl;
  LogTrace(metname) << debug.dumpTSOS(tsosAfterRefit);

  /*
  // -- 1st attempt
  if( filter()->isCompatibilitySatisfied() ) {
    if( filter()->layers().size() )   //  OR   if( filter()->goodState() ) ???  Maybe when only RPC hits are used...
      LogTrace(metname) << debug.dumpLayer( filter()->lastDetLayer() );
    else {
      LogTrace(metname) << "Compatibility satisfied, but all RecHits are invalid! A trajectory with only invalid hits will be loaded!" << endl;
      trajectoryContainer.push_back(new Trajectory(trajectoryFW));
      return trajectoryContainer;
    }
  }
  else {
    LogTrace(metname) << "Compatibility NOT satisfied after Forward filter! No trajectory will be loaded!" << endl;
    LogTrace(metname) << "Total chambers: " << filter()->getTotalCompatibleChambers() << "; DT: " 
		      << filter()->getDTCompatibleChambers() << "; CSC: " << filter()->getCSCCompatibleChambers() << endl;
    return trajectoryContainer; 
  }
  // -- end 1st attempt
  */

  // -- 2nd attempt
  if (filter()->goodState()) {
    LogTrace(metname) << debug.dumpLayer(filter()->lastDetLayer());
  } else if (filter()->isCompatibilitySatisfied()) {
    int foundValidRh = trajectoryFW.foundHits();  // number of valid hits
    LogTrace(metname) << "Compatibility satisfied after Forward filter, but too few valid RecHits (" << foundValidRh
                      << ")." << endl
                      << "A trajectory with only invalid RecHits will be loaded!" << endl;
    if (!foundValidRh) {
      trajectoryContainer.push_back(std::make_unique<Trajectory>(trajectoryFW));
      return trajectoryContainer;
    }
    Trajectory defaultTraj(seed, fwDirection);
    filter()->createDefaultTrajectory(trajectoryFW, defaultTraj);
    trajectoryContainer.push_back(std::make_unique<Trajectory>(defaultTraj));
    return trajectoryContainer;
  } else {
    LogTrace(metname) << "Compatibility NOT satisfied after Forward filter! No trajectory will be loaded!" << endl;
    LogTrace(metname) << "Total compatible chambers: " << filter()->getTotalCompatibleChambers()
                      << ";  DT: " << filter()->getDTCompatibleChambers()
                      << ";  CSC: " << filter()->getCSCCompatibleChambers()
                      << ";  RPC: " << filter()->getRPCCompatibleChambers()
                      << ";  GEM: " << filter()->getGEMCompatibleChambers()
                      << ";  ME0: " << filter()->getME0CompatibleChambers() << endl;
    return trajectoryContainer;
  }
  // -- end 2nd attempt

  LogTrace(metname) << "Number of DT/CSC/RPC/GEM/ME0 chamber used (fw): " << filter()->getDTChamberUsed() << "/"
                    << filter()->getCSCChamberUsed() << "/" << filter()->getRPCChamberUsed() << "/"
                    << filter()->getGEMChamberUsed() << "/" << filter()->getME0ChamberUsed() << endl;
  LogTrace(metname) << "Momentum: " << tsosAfterRefit.freeState()->momentum();

  if (!doBackwardFilter) {
    LogTrace(metname) << "Only forward refit requested. No backward refit will be performed!" << endl;

    // ***** Refit of fwd step *****
    //    if (doRefit && !trajectoryFW.empty() && filter()->goodState()){    // CHECK!!! Can trajectoryFW really be empty at this point??? And goodState...?
    if (doRefit) {
      pair<bool, Trajectory> refitResult = refitter()->refit(trajectoryFW);
      if (refitResult.first) {
        trajectoryContainer.push_back(std::make_unique<Trajectory>(refitResult.second));
        LogTrace(metname) << "StandAloneMuonTrajectoryBuilder refit output " << endl;
        LogTrace(metname) << debug.dumpTSOS(refitResult.second.lastMeasurement().updatedState());
      } else
        trajectoryContainer.push_back(std::make_unique<Trajectory>(trajectoryFW));
    } else
      trajectoryContainer.push_back(std::make_unique<Trajectory>(trajectoryFW));

    LogTrace(metname) << "Trajectory saved" << endl;
    return trajectoryContainer;
  }

  // ***** Backward filtering *****

  TrajectorySeed seedForBW;

  if (theBWSeedType == "noSeed") {
  } else if (theBWSeedType == "fromFWFit") {
    PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(
        tsosAfterRefit, trajectoryFW.lastMeasurement().recHit()->geographicalId().rawId());

    edm::OwnVector<TrackingRecHit> recHitsContainer;
    PropagationDirection seedDirection = (theSeedPosition == recoMuon::in) ? oppositeToMomentum : alongMomentum;
    TrajectorySeed fwFit(seedTSOS, recHitsContainer, seedDirection);

    seedForBW = fwFit;
  } else if (theBWSeedType == "fromGenerator") {
    seedForBW = seed;
  } else
    LogWarning(metname) << "Wrong seed type for the backward filter!";

  PropagationDirection bwDirection = (theSeedPosition == recoMuon::in) ? oppositeToMomentum : alongMomentum;
  Trajectory trajectoryBW(seedForBW, bwDirection);

  // FIXME! under check!
  bwfilter()->refit(tsosAfterRefit, filter()->lastDetLayer(), trajectoryBW);

  // Get the last TSOS
  TrajectoryStateOnSurface tsosAfterBWRefit = bwfilter()->lastUpdatedTSOS();

  LogTrace(metname) << "StandAloneMuonTrajectoryBuilder BW filter output " << endl;
  LogTrace(metname) << debug.dumpTSOS(tsosAfterBWRefit);

  LogTrace(metname) << "Number of RecHits: " << trajectoryBW.foundHits() << "\n"
                    << "Number of DT/CSC/RPC/GEM/ME0 chamber used (bw): " << bwfilter()->getDTChamberUsed() << "/"
                    << bwfilter()->getCSCChamberUsed() << "/" << bwfilter()->getRPCChamberUsed() << "/"
                    << bwfilter()->getGEMChamberUsed() << "/" << bwfilter()->getME0ChamberUsed();

  // -- The trajectory is "good" if there are at least 2 chambers used in total and at
  //    least 1 is "tracking" (DT or CSC)
  // -- The trajectory satisfies the "compatibility" requirements if there are at least
  //    2 compatible chambers (not necessarily used!) in total and at
  //    least 1 is "tracking" (DT or CSC)
  // 1st attempt
  /*
  if (bwfilter()->isCompatibilitySatisfied()) {
    
    if (doRefit && !trajectoryBW.empty() && bwfilter()->goodState()){
      pair<bool,Trajectory> refitResult = refitter()->refit(trajectoryBW);
      if (refitResult.first){
     	trajectoryContainer.push_back(new Trajectory(refitResult.second));
	LogTrace(metname) << "StandAloneMuonTrajectoryBuilder Refit output " << endl;
	LogTrace(metname) << debug.dumpTSOS(refitResult.second.lastMeasurement().updatedState());
      }
      else
	trajectoryContainer.push_back(new Trajectory(trajectoryBW));
    }
    else
      trajectoryContainer.push_back(new Trajectory(trajectoryBW));
    
    LogTrace(metname)<< "Trajectory saved" << endl;
    
  }
  //if the trajectory is not saved, but at least two chamber are used in the
  //forward filtering, try to build a new trajectory starting from the old
  //trajectory w/o the latest measurement and a looser chi2 cut
  else if ( filter()->getTotalChamberUsed() >= 2 ) {
    LogTrace(metname)<< "Trajectory NOT saved. Second Attempt." << endl;
    // FIXME:
    // a better choice could be: identify the poorest one, exclude it, redo
    // the fw and bw filtering. Or maybe redo only the bw without the excluded
    // measure. As first step I will port the ORCA algo, then I will move to the
    // above pattern.
    
  }

  else {
    LogTrace(metname) << "Compatibility NOT satisfied after Backward filter!" << endl;
    LogTrace(metname) << "The result of Forward filter will be loaded!" << endl;

    trajectoryContainer.push_back(new Trajectory(trajectoryFW));
  }
  */
  // end 1st attempt

  // 2nd attempt
  if (bwfilter()->goodState()) {
    LogTrace(metname) << debug.dumpLayer(bwfilter()->lastDetLayer());
  } else if (bwfilter()->isCompatibilitySatisfied()) {
    LogTrace(metname) << "Compatibility satisfied after Backward filter, but too few valid RecHits ("
                      << trajectoryBW.foundHits() << ")." << endl
                      << "The (good) result of FW filter will be loaded!" << endl;
    trajectoryContainer.push_back(std::make_unique<Trajectory>(trajectoryFW));
    return trajectoryContainer;
  } else {
    LogTrace(metname) << "Compatibility NOT satisfied after Backward filter!" << endl
                      << "The Forward trajectory will be invalidated and then loaded!" << endl;
    Trajectory defaultTraj(seed, fwDirection);
    filter()->createDefaultTrajectory(trajectoryFW, defaultTraj);
    trajectoryContainer.push_back(std::make_unique<Trajectory>(defaultTraj));
    return trajectoryContainer;
  }
  // end 2nd attempt

  if (doRefit) {
    pair<bool, Trajectory> refitResult = refitter()->refit(trajectoryBW);
    if (refitResult.first) {
      trajectoryContainer.push_back(std::make_unique<Trajectory>(refitResult.second));
      LogTrace(metname) << "StandAloneMuonTrajectoryBuilder Refit output " << endl;
      LogTrace(metname) << debug.dumpTSOS(refitResult.second.lastMeasurement().updatedState());
    } else
      trajectoryContainer.push_back(std::make_unique<Trajectory>(trajectoryBW));
  } else
    trajectoryContainer.push_back(std::make_unique<Trajectory>(trajectoryBW));

  LogTrace(metname) << "Trajectory saved" << endl;

  return trajectoryContainer;
}

StandAloneMuonTrajectoryBuilder::DetLayerWithState StandAloneMuonTrajectoryBuilder::propagateTheSeedTSOS(
    TrajectoryStateOnSurface& aTSOS, DetId& aDetId) {
  const std::string metname = "Muon|RecoMuon|StandAloneMuonTrajectoryBuilder";
  MuonPatternRecoDumper debug;

  DetId seedDetId(aDetId);
  //  const GeomDet* gdet = theService->trackingGeometry()->idToDet( seedDetId );

  TrajectoryStateOnSurface initialState(aTSOS);

  LogTrace(metname) << "Seed's Pt: " << initialState.freeState()->momentum().perp() << endl;

  LogTrace(metname) << "Seed's id: " << endl;
  LogTrace(metname) << debug.dumpMuonId(seedDetId);

  // Get the layer on which the seed relies
  const DetLayer* initialLayer = theService->detLayerGeometry()->idToLayer(seedDetId);

  LogTrace(metname) << "Seed's detLayer: " << endl;
  LogTrace(metname) << debug.dumpLayer(initialLayer);

  LogTrace(metname) << "TrajectoryStateOnSurface before propagation:" << endl;
  LogTrace(metname) << debug.dumpTSOS(initialState);

  PropagationDirection detLayerOrder = (theSeedPosition == recoMuon::in) ? oppositeToMomentum : alongMomentum;

  // ask for compatible layers
  vector<const DetLayer*> detLayers;

  if (theNavigationType == "Standard")
    detLayers =
        theService->muonNavigationSchool()->compatibleLayers(*initialLayer, *initialState.freeState(), detLayerOrder);
  else if (theNavigationType == "Direct") {
    DirectMuonNavigation navigation(&*theService->detLayerGeometry());
    detLayers = navigation.compatibleLayers(*initialState.freeState(), detLayerOrder);
  } else
    edm::LogError(metname) << "No Properly Navigation Selected!!" << endl;

  LogTrace(metname) << "There are " << detLayers.size() << " compatible layers" << endl;

  DetLayerWithState result = DetLayerWithState(initialLayer, initialState);

  if (!detLayers.empty()) {
    LogTrace(metname) << "Compatible layers:" << endl;
    for (vector<const DetLayer*>::const_iterator layer = detLayers.begin(); layer != detLayers.end(); layer++) {
      LogTrace(metname) << debug.dumpMuonId((*layer)->basicComponents().front()->geographicalId());
      LogTrace(metname) << debug.dumpLayer(*layer);
    }

    const DetLayer* finalLayer = detLayers.back();

    if (theSeedPosition == recoMuon::in)
      LogTrace(metname) << "Most internal one:" << endl;
    else
      LogTrace(metname) << "Most external one:" << endl;

    LogTrace(metname) << debug.dumpLayer(finalLayer);

    const TrajectoryStateOnSurface propagatedState =
        theService->propagator(theSeedPropagatorName)->propagate(initialState, finalLayer->surface());

    if (propagatedState.isValid()) {
      result = DetLayerWithState(finalLayer, propagatedState);

      LogTrace(metname) << "Trajectory State on Surface after the extrapolation" << endl;
      LogTrace(metname) << debug.dumpTSOS(propagatedState);
      LogTrace(metname) << debug.dumpLayer(finalLayer);
    } else
      LogTrace(metname) << "Extrapolation failed. Keep the original seed state" << endl;
  } else
    LogTrace(metname) << "No compatible layers. Keep the original seed state" << endl;

  return result;
}
