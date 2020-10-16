/**
 *  Class: GlobalTrajectoryBuilderBase
 *
 *   Base class for GlobalMuonTrajectoryBuilder and L3MuonTrajectoryBuilder
 *   Provide common tools and interface to reconstruct muons starting
 *   from a muon track reconstructed
 *   in the standalone muon system (with DT, CSC and RPC
 *   information).
 *   It tries to reconstruct the corresponding
 *   track in the tracker and performs
 *   matching between the reconstructed tracks
 *   in the muon system and the tracker.
 *
 *  \author N. Neumeister        Purdue University
 *  \author C. Liu               Purdue University
 *  \author A. Everett           Purdue University
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <algorithm>
#include <limits>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

//----------------
// Constructors --
//----------------
GlobalTrajectoryBuilderBase::GlobalTrajectoryBuilderBase(const edm::ParameterSet& par,
                                                         const MuonServiceProxy* service,
                                                         edm::ConsumesCollector& iC)
    : theLayerMeasurements(nullptr),
      theTrackTransformer(nullptr),
      theRegionBuilder(nullptr),
      theService(service),
      theGlbRefitter(nullptr) {
  theCategory =
      par.getUntrackedParameter<std::string>("Category", "Muon|RecoMuon|GlobalMuon|GlobalTrajectoryBuilderBase");

  edm::ParameterSet trackMatcherPSet = par.getParameter<edm::ParameterSet>("GlobalMuonTrackMatcher");
  theTrackMatcher = new GlobalMuonTrackMatcher(trackMatcherPSet, theService);

  theTrackerPropagatorName = par.getParameter<std::string>("TrackerPropagator");

  edm::ParameterSet trackTransformerPSet = par.getParameter<edm::ParameterSet>("TrackTransformer");
  theTrackTransformer = new TrackTransformer(trackTransformerPSet);

  edm::ParameterSet regionBuilderPSet = par.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");

  theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet, iC);

  // TrackRefitter parameters
  edm::ParameterSet refitterParameters = par.getParameter<edm::ParameterSet>("GlbRefitterParameters");
  theGlbRefitter = new GlobalMuonRefitter(refitterParameters, theService, iC);

  theMuonHitsOption = refitterParameters.getParameter<int>("MuonHitsOption");
  theRefitFlag = refitterParameters.getParameter<bool>("RefitFlag");

  theTrackerRecHitBuilderToken =
      iC.esConsumes(edm::ESInputTag("", par.getParameter<std::string>("TrackerRecHitBuilder")));
  theMuonRecHitBuilderToken = iC.esConsumes(edm::ESInputTag("", par.getParameter<std::string>("MuonRecHitBuilder")));
  theTopoToken = iC.esConsumes();

  theRPCInTheFit = par.getParameter<bool>("RefitRPCHits");

  theTECxScale = par.getParameter<double>("ScaleTECxFactor");
  theTECyScale = par.getParameter<double>("ScaleTECyFactor");
  thePtCut = par.getParameter<double>("PtCut");
  thePCut = par.getParameter<double>("PCut");
}

//--------------
// Destructor --
//--------------
GlobalTrajectoryBuilderBase::~GlobalTrajectoryBuilderBase() {
  if (theTrackMatcher)
    delete theTrackMatcher;
  if (theRegionBuilder)
    delete theRegionBuilder;
  if (theTrackTransformer)
    delete theTrackTransformer;
  if (theGlbRefitter)
    delete theGlbRefitter;
}

//
// set Event
//
void GlobalTrajectoryBuilderBase::setEvent(const edm::Event& event) {
  theEvent = &event;

  theTrackTransformer->setServices(theService->eventSetup());
  theRegionBuilder->setEvent(event);

  theGlbRefitter->setEvent(event);
  theGlbRefitter->setServices(theService->eventSetup());

  theTrackerRecHitBuilder = &theService->eventSetup().getData(theTrackerRecHitBuilderToken);
  theMuonRecHitBuilder = &theService->eventSetup().getData(theMuonRecHitBuilderToken);

  //Retrieve tracker topology from geometry
  theTopo = &theService->eventSetup().getData(theTopoToken);
}

//
// build a combined tracker-muon trajectory
//
MuonCandidate::CandidateContainer GlobalTrajectoryBuilderBase::build(const TrackCand& staCand,
                                                                     MuonCandidate::CandidateContainer& tkTrajs) const {
  LogTrace(theCategory) << " Begin Build" << std::endl;

  // tracker trajectory should be built and refit before this point
  if (tkTrajs.empty())
    return CandidateContainer();

  // add muon hits and refit/smooth trajectories
  CandidateContainer refittedResult;
  ConstRecHitContainer muonRecHits = getTransientRecHits(*(staCand.second));

  // check order of muon measurements
  if ((muonRecHits.size() > 1) &&
      (muonRecHits.front()->globalPosition().mag() > muonRecHits.back()->globalPosition().mag())) {
    LogTrace(theCategory) << "   reverse order: ";
  }

  for (auto&& it : tkTrajs) {
    // cut on tracks with low momenta
    LogTrace(theCategory) << "   Track p and pT " << it->trackerTrack()->p() << " " << it->trackerTrack()->pt();
    if (it->trackerTrack()->p() < thePCut || it->trackerTrack()->pt() < thePtCut)
      continue;

    // If true we will run theGlbRefitter->refit from all hits
    if (theRefitFlag) {
      ConstRecHitContainer trackerRecHits;
      if (it->trackerTrack().isNonnull()) {
        trackerRecHits = getTransientRecHits(*it->trackerTrack());
      } else {
        LogDebug(theCategory) << "     NEED HITS FROM TRAJ";
      }

      // ToDo: Do we need the following ?:
      // check for single TEC RecHits in trajectories in the overalp region
      if (std::abs(it->trackerTrack()->eta()) > 0.95 && std::abs(it->trackerTrack()->eta()) < 1.15 &&
          it->trackerTrack()->pt() < 60) {
        if (theTECxScale < 0 || theTECyScale < 0)
          trackerRecHits = selectTrackerHits(trackerRecHits);
        else
          fixTEC(trackerRecHits, theTECxScale, theTECyScale);
      }

      RefitDirection recHitDir = checkRecHitsOrdering(trackerRecHits);
      if (recHitDir == outToIn)
        reverse(trackerRecHits.begin(), trackerRecHits.end());

      reco::TransientTrack tTT(it->trackerTrack(), &*theService->magneticField(), theService->trackingGeometry());
      TrajectoryStateOnSurface innerTsos = tTT.innermostMeasurementState();

      edm::RefToBase<TrajectorySeed> tmpSeed;
      if (it->trackerTrack()->seedRef().isAvailable())
        tmpSeed = it->trackerTrack()->seedRef();

      if (!innerTsos.isValid()) {
        LogTrace(theCategory) << "     inner Trajectory State is invalid. ";
        continue;
      }

      innerTsos.rescaleError(100.);

      TC refitted0, refitted1;
      std::unique_ptr<Trajectory> tkTrajectory;

      // tracker only track
      if (!(it->trackerTrajectory() && it->trackerTrajectory()->isValid())) {
        refitted0 = theTrackTransformer->transform(it->trackerTrack());
        if (!refitted0.empty())
          tkTrajectory = std::make_unique<Trajectory>(*(refitted0.begin()));
        else
          edm::LogWarning(theCategory) << "     Failed to load tracker track trajectory";
      } else
        tkTrajectory = it->releaseTrackerTrajectory();
      if (tkTrajectory)
        tkTrajectory->setSeedRef(tmpSeed);

      // full track with all muon hits using theGlbRefitter
      ConstRecHitContainer allRecHits = trackerRecHits;
      allRecHits.insert(allRecHits.end(), muonRecHits.begin(), muonRecHits.end());
      refitted1 = theGlbRefitter->refit(*it->trackerTrack(), tTT, allRecHits, theMuonHitsOption, theTopo);
      LogTrace(theCategory) << "     This track-sta refitted to " << refitted1.size() << " trajectories";

      std::unique_ptr<Trajectory> glbTrajectory1;
      if (!refitted1.empty())
        glbTrajectory1 = std::make_unique<Trajectory>(*(refitted1.begin()));
      else
        LogDebug(theCategory) << "     Failed to load global track trajectory 1";
      if (glbTrajectory1)
        glbTrajectory1->setSeedRef(tmpSeed);

      if (glbTrajectory1 && tkTrajectory) {
        refittedResult.emplace_back(std::make_unique<MuonCandidate>(
            std::move(glbTrajectory1), it->muonTrack(), it->trackerTrack(), std::move(tkTrajectory)));
      }
    } else {
      edm::RefToBase<TrajectorySeed> tmpSeed;
      if (it->trackerTrack()->seedRef().isAvailable())
        tmpSeed = it->trackerTrack()->seedRef();

      TC refitted0;
      std::unique_ptr<Trajectory> tkTrajectory;
      if (!(it->trackerTrajectory() && it->trackerTrajectory()->isValid())) {
        refitted0 = theTrackTransformer->transform(it->trackerTrack());
        if (!refitted0.empty()) {
          tkTrajectory = std::make_unique<Trajectory>(*(refitted0.begin()));
        } else
          edm::LogWarning(theCategory) << "     Failed to load tracker track trajectory";
      } else
        tkTrajectory = it->releaseTrackerTrajectory();
      std::unique_ptr<Trajectory> cpy;
      if (tkTrajectory) {
        tkTrajectory->setSeedRef(tmpSeed);
        cpy = std::make_unique<Trajectory>(*tkTrajectory);
      }
      // Creating MuonCandidate using only the tracker trajectory:
      refittedResult.emplace_back(std::make_unique<MuonCandidate>(
          std::move(tkTrajectory), it->muonTrack(), it->trackerTrack(), std::move(cpy)));
    }
  }

  // choose the best global fit for this Standalone Muon based on the track probability
  CandidateContainer selectedResult;
  std::unique_ptr<MuonCandidate> tmpCand;
  double minProb = std::numeric_limits<double>::max();

  for (auto&& cand : refittedResult) {
    double prob = trackProbability(*cand->trajectory());
    LogTrace(theCategory) << "   refitted-track-sta with pT " << cand->trackerTrack()->pt() << " has probability "
                          << prob;

    if (prob < minProb or not tmpCand) {
      minProb = prob;
      tmpCand = std::move(cand);
    }
  }

  if (tmpCand)
    selectedResult.push_back(std::move(tmpCand));

  refittedResult.clear();

  return selectedResult;
}

//
// select tracker tracks within a region of interest
//
std::vector<GlobalTrajectoryBuilderBase::TrackCand> GlobalTrajectoryBuilderBase::chooseRegionalTrackerTracks(
    const TrackCand& staCand, const std::vector<TrackCand>& tkTs) {
  // define eta-phi region
  RectangularEtaPhiTrackingRegion regionOfInterest = defineRegionOfInterest(staCand.second);

  // get region's etaRange and phiMargin
  //UNUSED:  PixelRecoRange<float> etaRange = regionOfInterest.etaRange();
  //UNUSED:  TkTrackingRegionsMargin<float> phiMargin = regionOfInterest.phiMargin();

  std::vector<TrackCand> result;

  double deltaR_max = 1.0;

  for (auto&& is : tkTs) {
    double deltaR_tmp = deltaR(static_cast<double>(regionOfInterest.direction().eta()),
                               static_cast<double>(regionOfInterest.direction().phi()),
                               is.second->eta(),
                               is.second->phi());

    // for each trackCand in region, add trajectory and add to result
    //if ( inEtaRange && inPhiRange ) {
    if (deltaR_tmp < deltaR_max) {
      TrackCand tmpCand = TrackCand(is);
      result.push_back(tmpCand);
    }
  }

  return result;
}

//
// define a region of interest within the tracker
//
RectangularEtaPhiTrackingRegion GlobalTrajectoryBuilderBase::defineRegionOfInterest(
    const reco::TrackRef& staTrack) const {
  std::unique_ptr<RectangularEtaPhiTrackingRegion> region1 = theRegionBuilder->region(staTrack);

  TkTrackingRegionsMargin<float> etaMargin(std::abs(region1->etaRange().min() - region1->etaRange().mean()),
                                           std::abs(region1->etaRange().max() - region1->etaRange().mean()));

  RectangularEtaPhiTrackingRegion region2(region1->direction(),
                                          region1->origin(),
                                          region1->ptMin(),
                                          region1->originRBound(),
                                          region1->originZBound(),
                                          etaMargin,
                                          region1->phiMargin());

  return region2;
}

//
// calculate the tail probability (-ln(P)) of a fit
//
double GlobalTrajectoryBuilderBase::trackProbability(const Trajectory& track) const {
  if (track.ndof() > 0 && track.chiSquared() > 0) {
    return -LnChiSquaredProbability(track.chiSquared(), track.ndof());
  } else {
    return 0.0;
  }
}

//
// print RecHits
//
void GlobalTrajectoryBuilderBase::printHits(const ConstRecHitContainer& hits) const {
  LogTrace(theCategory) << "Used RecHits: " << hits.size();
  for (auto&& ir : hits) {
    if (!ir->isValid()) {
      LogTrace(theCategory) << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = ir->globalPosition();

    LogTrace(theCategory) << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y()) << "  z = " << pos.z()
                          << "  dimension = " << ir->dimension() << "  " << ir->det()->geographicalId().det() << "  "
                          << ir->det()->subDetector();
  }
}

//
// check order of RechIts on a trajectory
//
GlobalTrajectoryBuilderBase::RefitDirection GlobalTrajectoryBuilderBase::checkRecHitsOrdering(
    const TransientTrackingRecHit::ConstRecHitContainer& recHits) const {
  if (!recHits.empty()) {
    ConstRecHitContainer::const_iterator frontHit = recHits.begin();
    ConstRecHitContainer::const_iterator backHit = recHits.end() - 1;
    while (!(*frontHit)->isValid() && frontHit != backHit) {
      frontHit++;
    }
    while (!(*backHit)->isValid() && backHit != frontHit) {
      backHit--;
    }

    double rFirst = (*frontHit)->globalPosition().mag();
    double rLast = (*backHit)->globalPosition().mag();

    if (rFirst < rLast)
      return inToOut;
    else if (rFirst > rLast)
      return outToIn;
    else {
      edm::LogError(theCategory) << "Impossible to determine the rechits order" << std::endl;
      return undetermined;
    }
  } else {
    edm::LogError(theCategory) << "Impossible to determine the rechits order" << std::endl;
    return undetermined;
  }
}

//
// select trajectories with only a single TEC hit
//
GlobalTrajectoryBuilderBase::ConstRecHitContainer GlobalTrajectoryBuilderBase::selectTrackerHits(
    const ConstRecHitContainer& all) const {
  int nTEC(0);

  ConstRecHitContainer hits;
  for (auto&& i : all) {
    if (!i->isValid())
      continue;
    if (i->det()->geographicalId().det() == DetId::Tracker &&
        i->det()->geographicalId().subdetId() == StripSubdetector::TEC) {
      nTEC++;
    } else {
      hits.push_back(i);
    }
    if (nTEC > 1)
      return all;
  }

  return hits;
}

//
// rescale errors of outermost TEC RecHit
//
void GlobalTrajectoryBuilderBase::fixTEC(ConstRecHitContainer& all, double scl_x, double scl_y) const {
  int nTEC(0);
  ConstRecHitContainer::iterator lone_tec;

  for (ConstRecHitContainer::iterator i = all.begin(); i != all.end(); i++) {
    if (!(*i)->isValid())
      continue;

    if ((*i)->det()->geographicalId().det() == DetId::Tracker &&
        (*i)->det()->geographicalId().subdetId() == StripSubdetector::TEC) {
      lone_tec = i;
      nTEC++;

      if ((i + 1) != all.end() && (*(i + 1))->isValid() &&
          (*(i + 1))->det()->geographicalId().det() == DetId::Tracker &&
          (*(i + 1))->det()->geographicalId().subdetId() == StripSubdetector::TEC) {
        nTEC++;
        break;
      }
    }

    if (nTEC > 1)
      break;
  }

  int hitDet = (*lone_tec)->hit()->geographicalId().det();
  int hitSubDet = (*lone_tec)->hit()->geographicalId().subdetId();
  if (nTEC == 1 && (*lone_tec)->hit()->isValid() && hitDet == DetId::Tracker && hitSubDet == StripSubdetector::TEC) {
    // rescale the TEC rechit error matrix in its rotated frame
    const SiStripRecHit2D* strip = dynamic_cast<const SiStripRecHit2D*>((*lone_tec)->hit());
    if (strip && strip->det()) {
      LocalPoint pos = strip->localPosition();
      if ((*lone_tec)->detUnit()) {
        const StripTopology* topology = dynamic_cast<const StripTopology*>(&(*lone_tec)->detUnit()->topology());
        if (topology) {
          // rescale the local error along/perp the strip by a factor
          float angle = topology->stripAngle(topology->strip((*lone_tec)->hit()->localPosition()));
          LocalError error = strip->localPositionError();
          LocalError rotError = error.rotate(angle);
          LocalError scaledError(rotError.xx() * scl_x * scl_x, 0, rotError.yy() * scl_y * scl_y);
          error = scaledError.rotate(-angle);
          /// freeze this hit, make sure it will not be recomputed during fitting
          //// the implemetantion below works with cloning
          //// to get a RecHitPointer to SiStripRecHit2D, the only  method that works is
          //// RecHitPointer MuonTransientTrackingRecHit::build(const GeomDet*,const TrackingRecHit*)
          SiStripRecHit2D* st = new SiStripRecHit2D(pos, error, *strip->det(), strip->cluster());
          *lone_tec = MuonTransientTrackingRecHit::build((*lone_tec)->det(), st);
        }
      }
    }
  }
}

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
//
// get transient RecHits
//
TransientTrackingRecHit::ConstRecHitContainer GlobalTrajectoryBuilderBase::getTransientRecHits(
    const reco::Track& track) const {
  TransientTrackingRecHit::ConstRecHitContainer result;

  TrajectoryStateOnSurface currTsos = trajectoryStateTransform::innerStateOnSurface(
      track, *theService->trackingGeometry(), &*theService->magneticField());

  auto tkbuilder = static_cast<TkTransientTrackingRecHitBuilder const*>(theTrackerRecHitBuilder);
  auto hitCloner = tkbuilder->cloner();
  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if ((*hit)->isValid()) {
      DetId recoid = (*hit)->geographicalId();
      if (recoid.det() == DetId::Tracker) {
        if (!(*hit)->hasPositionAndError()) {
          TrajectoryStateOnSurface predTsos =
              theService->propagator(theTrackerPropagatorName)
                  ->propagate(currTsos, theService->trackingGeometry()->idToDet(recoid)->surface());

          if (!predTsos.isValid()) {
            edm::LogError("MissingTransientHit")
                << "Could not get a tsos on the hit surface. We will miss a tracking hit.";
            continue;
          }
          currTsos = predTsos;
          auto h = (**hit).cloneForFit(*tkbuilder->geometry()->idToDet((**hit).geographicalId()));
          result.emplace_back(hitCloner.makeShared(h, predTsos));
        } else {
          result.push_back((*hit)->cloneSH());
        }
      } else if (recoid.det() == DetId::Muon) {
        if ((*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit) {
          LogDebug(theCategory) << "RPC Rec Hit discarded";
          continue;
        }
        result.push_back(theMuonRecHitBuilder->build(&**hit));
      }
    }
  }

  return result;
}
