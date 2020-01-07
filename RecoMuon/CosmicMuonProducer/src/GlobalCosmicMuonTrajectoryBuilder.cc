/**
 *  Class: GlobalCosmicMuonTrajectoryBuilder
 *
 *  \author Chang Liu  -  Purdue University <Chang.Liu@cern.ch>
 *
 **/
#include "RecoMuon/CosmicMuonProducer/interface/GlobalCosmicMuonTrajectoryBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace edm;

//
// constructor
//
GlobalCosmicMuonTrajectoryBuilder::GlobalCosmicMuonTrajectoryBuilder(const edm::ParameterSet& par,
                                                                     const MuonServiceProxy* service,
                                                                     edm::ConsumesCollector& iC)
    : theService(service) {
  ParameterSet smootherPSet = par.getParameter<ParameterSet>("SmootherParameters");
  theSmoother = new CosmicMuonSmoother(smootherPSet, theService);

  ParameterSet trackMatcherPSet = par.getParameter<ParameterSet>("GlobalMuonTrackMatcher");
  theTrackMatcher = new GlobalMuonTrackMatcher(trackMatcherPSet, theService);

  theTkTrackToken = iC.consumes<reco::TrackCollection>(par.getParameter<InputTag>("TkTrackCollectionLabel"));

  theTrackerRecHitBuilderName = par.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = par.getParameter<string>("MuonRecHitBuilder");
  thePropagatorName = par.getParameter<string>("Propagator");
  category_ = "Muon|RecoMuon|CosmicMuon|GlobalCosmicMuonTrajectoryBuilder";
}

//
// destructor
//

GlobalCosmicMuonTrajectoryBuilder::~GlobalCosmicMuonTrajectoryBuilder() {
  if (theSmoother)
    delete theSmoother;
  if (theTrackMatcher)
    delete theTrackMatcher;
}

//
// set Event
//
void GlobalCosmicMuonTrajectoryBuilder::setEvent(const edm::Event& event) {
  event.getByToken(theTkTrackToken, theTrackerTracks);

  //  edm::Handle<std::vector<Trajectory> > handleTrackerTrajs;
  //  if ( event.getByLabel(theTkTrackLabel,handleTrackerTrajs) && handleTrackerTrajs.isValid() ) {
  //      tkTrajsAvailable = true;
  //      allTrackerTrajs = &*handleTrackerTrajs;
  //      LogInfo("GlobalCosmicMuonTrajectoryBuilder")
  //	<< "Tk Trajectories Found! " << endl;
  //  } else {
  //      LogInfo("GlobalCosmicMuonTrajectoryBuilder")
  //	<< "No Tk Trajectories Found! " << endl;
  //      tkTrajsAvailable = false;
  //  }

  theService->eventSetup().get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName, theTrackerRecHitBuilder);
  theService->eventSetup().get<TransientRecHitRecord>().get(theMuonRecHitBuilderName, theMuonRecHitBuilder);
}

//
// reconstruct trajectories
//
MuonCandidate::CandidateContainer GlobalCosmicMuonTrajectoryBuilder::trajectories(const TrackCand& muCand) {
  MuonCandidate::CandidateContainer result;

  if (!theTrackerTracks.isValid()) {
    LogTrace(category_) << "Tracker Track collection is invalid!!!";
    return result;
  }

  LogTrace(category_) << "Found " << theTrackerTracks->size() << " tracker Tracks";
  if (theTrackerTracks->empty())
    return result;

  vector<TrackCand> matched = match(muCand, theTrackerTracks);

  LogTrace(category_) << "TrackMatcher found " << matched.size() << "tracker tracks matched";

  if (matched.empty())
    return result;
  reco::TrackRef tkTrack = matched.front().second;

  if (tkTrack.isNull())
    return result;
  reco::TrackRef muTrack = muCand.second;

  ConstRecHitContainer muRecHits;

  if (muCand.first == nullptr || !muCand.first->isValid()) {
    muRecHits = getTransientRecHits(*muTrack);
  } else {
    muRecHits = muCand.first->recHits();
  }

  LogTrace(category_) << "mu RecHits: " << muRecHits.size();

  ConstRecHitContainer tkRecHits = getTransientRecHits(*tkTrack);

  //  if ( !tkTrajsAvailable ) {
  //     tkRecHits = getTransientRecHits(*tkTrack);
  //  } else {
  //     tkRecHits = allTrackerTrajs->front().recHits();
  //  }

  ConstRecHitContainer hits;  //= tkRecHits;
  LogTrace(category_) << "tk RecHits: " << tkRecHits.size();

  //  hits.insert(hits.end(), muRecHits.begin(), muRecHits.end());
  //  stable_sort(hits.begin(), hits.end(), DecreasingGlobalY());

  sortHits(hits, muRecHits, tkRecHits);

  //  LogTrace(category_)<< "Used RecHits after sort: "<<hits.size()<<endl;;
  //  LogTrace(category_) <<utilities()->print(hits)<<endl;
  //  LogTrace(category_) << "== End of Used RecHits == "<<endl;

  TrajectoryStateOnSurface muonState1 = trajectoryStateTransform::innerStateOnSurface(
      *muTrack, *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface tkState1 = trajectoryStateTransform::innerStateOnSurface(
      *tkTrack, *theService->trackingGeometry(), &*theService->magneticField());

  TrajectoryStateOnSurface muonState2 = trajectoryStateTransform::outerStateOnSurface(
      *muTrack, *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface tkState2 = trajectoryStateTransform::outerStateOnSurface(
      *tkTrack, *theService->trackingGeometry(), &*theService->magneticField());

  TrajectoryStateOnSurface firstState1 =
      (muonState1.globalPosition().y() > tkState1.globalPosition().y()) ? muonState1 : tkState1;
  TrajectoryStateOnSurface firstState2 =
      (muonState2.globalPosition().y() > tkState2.globalPosition().y()) ? muonState2 : tkState2;

  TrajectoryStateOnSurface firstState =
      (firstState1.globalPosition().y() > firstState2.globalPosition().y()) ? firstState1 : firstState2;

  GlobalPoint front, back;
  if (!hits.empty()) {
    front = hits.front()->globalPosition();
    back = hits.back()->globalPosition();
    if ((front.perp() < 130 && fabs(front.z()) < 300) || (back.perp() < 130 && fabs(back.z()) < 300)) {
      if (hits.front()->globalPosition().perp() > hits.back()->globalPosition().perp())
        reverse(hits.begin(), hits.end());
      tkState1 = trajectoryStateTransform::innerStateOnSurface(
          *tkTrack, *theService->trackingGeometry(), &*theService->magneticField());
      tkState2 = trajectoryStateTransform::outerStateOnSurface(
          *tkTrack, *theService->trackingGeometry(), &*theService->magneticField());
      firstState = (tkState1.globalPosition().perp() < tkState2.globalPosition().perp()) ? tkState1 : tkState2;
    }
  }
  if (!firstState.isValid())
    return result;
  LogTrace(category_) << "firstTSOS pos: " << firstState.globalPosition() << "mom: " << firstState.globalMomentum();

  // begin refitting

  TrajectorySeed seed;
  vector<Trajectory> refitted = theSmoother->trajectories(seed, hits, firstState);

  if (refitted.empty()) {
    LogTrace(category_) << "smoothing trajectories fail";
    refitted = theSmoother->fit(seed, hits, firstState);  //FIXME
  }

  if (refitted.empty()) {
    LogTrace(category_) << "refit fail";
    return result;
  }

  auto myTraj = std::make_unique<Trajectory>(refitted.front());

  const std::vector<TrajectoryMeasurement>& mytms = myTraj->measurements();
  LogTrace(category_) << "measurements in final trajectory " << mytms.size();
  LogTrace(category_) << "Orignally there are " << tkTrack->found() << " tk rhs and " << muTrack->found() << " mu rhs.";

  if (mytms.size() <= tkTrack->found()) {
    LogTrace(category_) << "insufficient measurements. skip... ";
    return result;
  }

  result.push_back(std::make_unique<MuonCandidate>(std::move(myTraj), muTrack, tkTrack));
  LogTrace(category_) << "final global cosmic muon: ";
  for (std::vector<TrajectoryMeasurement>::const_iterator itm = mytms.begin(); itm != mytms.end(); ++itm) {
    LogTrace(category_) << "updated pos " << itm->updatedState().globalPosition() << "mom "
                        << itm->updatedState().globalMomentum();
  }
  return result;
}

void GlobalCosmicMuonTrajectoryBuilder::sortHits(ConstRecHitContainer& hits,
                                                 ConstRecHitContainer& muonHits,
                                                 ConstRecHitContainer& tkHits) {
  if (tkHits.empty()) {
    LogTrace(category_) << "No valid tracker hits";
    return;
  }
  if (muonHits.empty()) {
    LogTrace(category_) << "No valid muon hits";
    return;
  }

  ConstRecHitContainer::const_iterator frontTkHit = tkHits.begin();
  ConstRecHitContainer::const_iterator backTkHit = tkHits.end() - 1;
  while (!(*frontTkHit)->isValid() && frontTkHit != backTkHit) {
    frontTkHit++;
  }
  while (!(*backTkHit)->isValid() && backTkHit != frontTkHit) {
    backTkHit--;
  }

  ConstRecHitContainer::const_iterator frontMuHit = muonHits.begin();
  ConstRecHitContainer::const_iterator backMuHit = muonHits.end() - 1;
  while (!(*frontMuHit)->isValid() && frontMuHit != backMuHit) {
    frontMuHit++;
  }
  while (!(*backMuHit)->isValid() && backMuHit != frontMuHit) {
    backMuHit--;
  }

  if (frontTkHit == backTkHit) {
    LogTrace(category_) << "No valid tracker hits";
    return;
  }
  if (frontMuHit == backMuHit) {
    LogTrace(category_) << "No valid muon hits";
    return;
  }

  GlobalPoint frontTkPos = (*frontTkHit)->globalPosition();
  GlobalPoint backTkPos = (*backTkHit)->globalPosition();

  GlobalPoint frontMuPos = (*frontMuHit)->globalPosition();
  GlobalPoint backMuPos = (*backMuHit)->globalPosition();

  //sort hits going from higher to lower positions
  if (frontTkPos.y() < backTkPos.y()) {  //check if tk hits order same direction
    reverse(tkHits.begin(), tkHits.end());
  }

  if (frontMuPos.y() < backMuPos.y()) {
    reverse(muonHits.begin(), muonHits.end());
  }

  //  LogTrace(category_)<< "tkHits after sort: "<<tkHits.size()<<endl;;
  //  LogTrace(category_) <<utilities()->print(tkHits)<<endl;
  //  LogTrace(category_) << "== End of tkHits == "<<endl;

  //  LogTrace(category_)<< "muonHits after sort: "<<muonHits.size()<<endl;;
  //  LogTrace(category_) <<utilities()->print(muonHits)<<endl;
  //  LogTrace(category_)<< "== End of muonHits == "<<endl;

  //separate muon hits into 2 different hemisphere
  ConstRecHitContainer::iterator middlepoint = muonHits.begin();
  bool insertInMiddle = false;

  for (ConstRecHitContainer::iterator ihit = muonHits.begin(); ihit != muonHits.end() - 1; ihit++) {
    GlobalPoint ipos = (*ihit)->globalPosition();
    GlobalPoint nextpos = (*(ihit + 1))->globalPosition();
    if ((ipos - nextpos).mag() < 100.0)
      continue;

    GlobalPoint middle((ipos.x() + nextpos.x()) / 2, (ipos.y() + nextpos.y()) / 2, (ipos.z() + nextpos.z()) / 2);
    LogTrace(category_) << "ipos " << ipos << "nextpos" << nextpos << " middle " << middle << endl;
    if ((middle.perp() < ipos.perp()) && (middle.perp() < nextpos.perp())) {
      LogTrace(category_) << "found middlepoint" << endl;
      middlepoint = ihit;
      insertInMiddle = true;
      break;
    }
  }

  //insert track hits in correct order
  if (insertInMiddle) {  //if tk hits should be sandwich
    GlobalPoint jointpointpos = (*middlepoint)->globalPosition();
    LogTrace(category_) << "jointpoint " << jointpointpos << endl;
    if ((frontTkPos - jointpointpos).mag() >
        (backTkPos - jointpointpos).mag()) {  //check if tk hits order same direction
      reverse(tkHits.begin(), tkHits.end());
    }
    muonHits.insert(middlepoint + 1, tkHits.begin(), tkHits.end());
    hits = muonHits;
  } else {                                  // append at one end
    if (frontTkPos.y() < frontMuPos.y()) {  //insert at the end
      LogTrace(category_) << "insert at the end " << frontTkPos << frontMuPos << endl;

      hits = muonHits;
      hits.insert(hits.end(), tkHits.begin(), tkHits.end());
    } else {  //insert at the beginning
      LogTrace(category_) << "insert at the beginning " << frontTkPos << frontMuPos << endl;
      hits = tkHits;
      hits.insert(hits.end(), muonHits.begin(), muonHits.end());
    }
  }
}

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

TransientTrackingRecHit::ConstRecHitContainer GlobalCosmicMuonTrajectoryBuilder::getTransientRecHits(
    const reco::Track& track) const {
  TransientTrackingRecHit::ConstRecHitContainer result;

  auto hitCloner = static_cast<TkTransientTrackingRecHitBuilder const*>(theTrackerRecHitBuilder.product())->cloner();
  TrajectoryStateOnSurface currTsos = trajectoryStateTransform::innerStateOnSurface(
      track, *theService->trackingGeometry(), &*theService->magneticField());
  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if ((*hit)->isValid()) {
      DetId recoid = (*hit)->geographicalId();
      if (recoid.det() == DetId::Tracker) {
        TrajectoryStateOnSurface predTsos =
            theService->propagator(thePropagatorName)
                ->propagate(currTsos, theService->trackingGeometry()->idToDet(recoid)->surface());
        LogTrace(category_) << "predtsos " << predTsos.isValid();
        if (predTsos.isValid()) {
          currTsos = predTsos;
          result.emplace_back(hitCloner(**hit, predTsos));
        }
      } else if (recoid.det() == DetId::Muon) {
        result.push_back(theMuonRecHitBuilder->build(&**hit));
      }
    }
  }
  return result;
}

std::vector<GlobalCosmicMuonTrajectoryBuilder::TrackCand> GlobalCosmicMuonTrajectoryBuilder::match(
    const TrackCand& mu, const edm::Handle<reco::TrackCollection>& tktracks) {
  std::vector<TrackCand> result;

  TrajectoryStateOnSurface innerTsos = trajectoryStateTransform::innerStateOnSurface(
      *(mu.second), *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface outerTsos = trajectoryStateTransform::outerStateOnSurface(
      *(mu.second), *theService->trackingGeometry(), &*theService->magneticField());
  //build tracker TrackCands and pick the best match if size greater than 2
  vector<TrackCand> tkTrackCands;
  for (reco::TrackCollection::size_type i = 0; i < theTrackerTracks->size(); ++i) {
    reco::TrackRef tkTrack(theTrackerTracks, i);
    TrackCand tkCand = TrackCand((Trajectory*)nullptr, tkTrack);
    tkTrackCands.push_back(tkCand);
    LogTrace(category_) << "chisq is " << theTrackMatcher->match(mu, tkCand, 0, 0);
    LogTrace(category_) << "d is " << theTrackMatcher->match(mu, tkCand, 1, 0);
    LogTrace(category_) << "r_pos is " << theTrackMatcher->match(mu, tkCand, 2, 0);
  }

  // now if only 1 tracker tracks, return it
  if (tkTrackCands.size() <= 1) {
    return tkTrackCands;
  }

  // if there're many tracker tracks

  // if muon is only on one side
  GlobalPoint innerPos = innerTsos.globalPosition();
  GlobalPoint outerPos = outerTsos.globalPosition();

  if ((innerPos.basicVector().dot(innerTsos.globalMomentum().basicVector()) *
           outerPos.basicVector().dot(outerTsos.globalMomentum().basicVector()) >
       0)) {
    GlobalPoint geoInnerPos = (innerPos.mag() < outerPos.mag()) ? innerPos : outerPos;
    LogTrace(category_) << "geoInnerPos Mu " << geoInnerPos << endl;

    // if there're tracker tracks totally on the other half
    // and there're tracker tracks on the same half
    // remove the tracks on the other half
    for (vector<TrackCand>::const_iterator itkCand = tkTrackCands.begin(); itkCand != tkTrackCands.end(); ++itkCand) {
      reco::TrackRef tkTrack = itkCand->second;

      GlobalPoint tkInnerPos(tkTrack->innerPosition().x(), tkTrack->innerPosition().y(), tkTrack->innerPosition().z());
      GlobalPoint tkOuterPos(tkTrack->outerPosition().x(), tkTrack->outerPosition().y(), tkTrack->outerPosition().z());
      LogTrace(category_) << "tkTrack " << tkInnerPos << " " << tkOuterPos << endl;

      float closetDistance11 = (geoInnerPos - tkInnerPos).mag();
      float closetDistance12 = (geoInnerPos - tkOuterPos).mag();
      float closetDistance1 = (closetDistance11 < closetDistance12) ? closetDistance11 : closetDistance12;
      LogTrace(category_) << "closetDistance1 " << closetDistance1 << endl;

      if (true || !isTraversing(*tkTrack)) {
        bool keep = true;
        for (vector<TrackCand>::const_iterator itkCand2 = tkTrackCands.begin(); itkCand2 != tkTrackCands.end();
             ++itkCand2) {
          if (itkCand2 == itkCand)
            continue;
          reco::TrackRef tkTrack2 = itkCand2->second;

          GlobalPoint tkInnerPos2(
              tkTrack2->innerPosition().x(), tkTrack2->innerPosition().y(), tkTrack2->innerPosition().z());
          GlobalPoint tkOuterPos2(
              tkTrack2->outerPosition().x(), tkTrack2->outerPosition().y(), tkTrack2->outerPosition().z());
          LogTrace(category_) << "tkTrack2 " << tkInnerPos2 << " " << tkOuterPos2 << endl;

          float farthestDistance21 = (geoInnerPos - tkInnerPos2).mag();
          float farthestDistance22 = (geoInnerPos - tkOuterPos2).mag();
          float farthestDistance2 = (farthestDistance21 > farthestDistance22) ? farthestDistance21 : farthestDistance22;
          LogTrace(category_) << "farthestDistance2 " << farthestDistance2 << endl;

          if (closetDistance1 > farthestDistance2 - 1e-3) {
            keep = false;
            break;
          }
        }
        if (keep)
          result.push_back(*itkCand);
        else
          LogTrace(category_) << "The Track is on different hemisphere" << endl;
      } else {
        result.push_back(*itkCand);
      }
    }
    if (result.empty()) {
      //if all tk tracks on the other side, still keep them
      result = tkTrackCands;
    }
  } else {  // muon is traversing
    result = tkTrackCands;
  }

  // match muCand to tkTrackCands
  vector<TrackCand> matched_trackerTracks = theTrackMatcher->match(mu, result);

  LogTrace(category_) << "TrackMatcher found " << matched_trackerTracks.size() << "tracker tracks matched";

  //now pick the best matched one
  if (matched_trackerTracks.size() < 2) {
    return matched_trackerTracks;
  } else {
    // in case of more than 1 tkTrack,
    // select the best-one based on distance (matchOption==1)
    // at innermost Mu hit surface. (surfaceOption == 0)
    result.clear();
    TrackCand bestMatch;

    double quality = 1e6;
    double max_quality = 1e6;
    for (vector<TrackCand>::const_iterator iter = matched_trackerTracks.begin(); iter != matched_trackerTracks.end();
         iter++) {
      quality = theTrackMatcher->match(mu, *iter, 1, 0);
      LogTrace(category_) << " quality of tracker track is " << quality;
      if (quality < max_quality) {
        max_quality = quality;
        bestMatch = (*iter);
      }
    }
    LogTrace(category_) << " Picked tracker track with quality " << max_quality;
    result.push_back(bestMatch);
    return result;
  }
}

bool GlobalCosmicMuonTrajectoryBuilder::isTraversing(const reco::Track& track) const {
  trackingRecHit_iterator firstValid;
  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if ((*hit)->isValid()) {
      firstValid = hit;
      break;
    }
  }

  trackingRecHit_iterator lastValid;
  for (trackingRecHit_iterator hit = track.recHitsEnd() - 1; hit != track.recHitsBegin() - 1; --hit) {
    if ((*hit)->isValid()) {
      lastValid = hit;
      break;
    }
  }

  GlobalPoint posFirst = theService->trackingGeometry()->idToDet((*firstValid)->geographicalId())->position();

  GlobalPoint posLast = theService->trackingGeometry()->idToDet((*lastValid)->geographicalId())->position();

  GlobalPoint middle(
      (posFirst.x() + posLast.x()) / 2, (posFirst.y() + posLast.y()) / 2, (posFirst.z() + posLast.z()) / 2);

  if ((middle.mag() < posFirst.mag()) && (middle.mag() < posLast.mag())) {
    return true;
  }
  return false;
}
