/*
 * \class TrackerSeedCleaner
 *  Reference class for seeds cleaning
 *  Seeds Cleaner based on sharedHits cleaning, direction cleaning and pt cleaning
    \author A. Grelli -  Purdue University, Pavia University
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

using namespace std;
using namespace edm;

//
// inizialization
//
void TrackerSeedCleaner::init(const MuonServiceProxy* service) { theProxyService = service; }

//
//
//
void TrackerSeedCleaner::setEvent(const edm::Event& event) { event.getByToken(beamspotToken_, bsHandle_); }

//
// clean seeds
//
void TrackerSeedCleaner::clean(const reco::TrackRef& muR,
                               const RectangularEtaPhiTrackingRegion& region,
                               tkSeeds& seeds) {
  // call the shared input cleaner
  if (cleanBySharedHits) {
    seeds = nonRedundantSeeds(seeds);
  }

  theTTRHBuilder = theProxyService->eventSetup().getHandle(theTTRHBuilderToken);

  LogDebug("TrackerSeedCleaner") << seeds.size() << " trajectory seeds to the events before cleaning" << endl;

  //check the validity otherwise vertexing
  const reco::BeamSpot& bs = *bsHandle_;
  /*reco track and seeds as arguments. Seeds eta and phi are checked and 
   based on deviation from L2 eta and phi seed is accepted or not*/

  std::vector<TrajectorySeed> result;

  TSCBLBuilderNoMaterial tscblBuilder;
  // PerigeeConversions tspConverter;
  for (TrajectorySeedCollection::iterator seed = seeds.begin(); seed < seeds.end(); ++seed) {
    if (seed->nHits() < 2)
      continue;
    //get parameters and errors from the seed state
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(seed->recHits().end() - 1));
    TrajectoryStateOnSurface state = trajectoryStateTransform::transientState(
        seed->startingState(), recHit->surface(), theProxyService->magneticField().product());

    TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed =
        tscblBuilder(*state.freeState(), bs);  //as in TrackProducerAlgorithms
    if (!tsAtClosestApproachSeed.isValid())
      continue;
    GlobalPoint vSeed1 = tsAtClosestApproachSeed.trackStateAtPCA().position();
    GlobalVector pSeed = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
    GlobalPoint vSeed(vSeed1.x() - bs.x0(), vSeed1.y() - bs.y0(), vSeed1.z() - bs.z0());

    //eta,phi info from seeds
    double etaSeed = state.globalMomentum().eta();
    double phiSeed = pSeed.phi();

    //if the limits are too stringent rescale limits
    typedef PixelRecoRange<float> Range;
    typedef TkTrackingRegionsMargin<float> Margin;

    Range etaRange = region.etaRange();
    double etaLimit = (fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) < 0.1)
                          ? 0.1
                          : fabs(fabs(etaRange.max()) - fabs(etaRange.mean()));

    Margin phiMargin = region.phiMargin();
    double phiLimit = (phiMargin.right() < 0.1) ? 0.1 : phiMargin.right();

    double ptSeed = pSeed.perp();
    double ptMin = (region.ptMin() > 3.5) ? 3.5 : region.ptMin();
    // Clean
    bool inEtaRange = etaSeed >= (etaRange.mean() - etaLimit) && etaSeed <= (etaRange.mean() + etaLimit);
    bool inPhiRange = (fabs(deltaPhi(phiSeed, double(region.direction().phi()))) < phiLimit);
    // pt cleaner
    bool inPtRange = ptSeed >= ptMin && ptSeed <= 2 * (muR->pt());

    // save efficiency don't clean triplets with pt cleaner
    if (seed->nHits() == 3)
      inPtRange = true;

    // use pt and angle cleaners
    if (inPtRange && usePt_Cleaner && !useDirection_Cleaner) {
      result.push_back(*seed);
      LogDebug("TrackerSeedCleaner") << " Keeping the seed : this seed passed pt selection";
    }

    // use only angle default option
    if (inEtaRange && inPhiRange && !usePt_Cleaner && useDirection_Cleaner) {
      result.push_back(*seed);
      LogDebug("TrackerSeedCleaner") << " Keeping the seed : this seed passed direction selection";
    }

    // use all the cleaners
    if (inEtaRange && inPhiRange && inPtRange && usePt_Cleaner && useDirection_Cleaner) {
      result.push_back(*seed);
      LogDebug("TrackerSeedCleaner") << " Keeping the seed : this seed passed direction and pt selection";
    }

    LogDebug("TrackerSeedCleaner") << " eta for current seed " << etaSeed << "\n"
                                   << " phi for current seed " << phiSeed << "\n"
                                   << " eta for L2 track  " << muR->eta() << "\n"
                                   << " phi for L2 track  " << muR->phi() << "\n";
  }

  //the new seeds collection
  if (!result.empty() && (useDirection_Cleaner || usePt_Cleaner))
    seeds.swap(result);

  LogDebug("TrackerSeedCleaner") << seeds.size() << " trajectory seeds to the events after cleaning" << endl;

  return;
}

TrackerSeedCleaner::tkSeeds TrackerSeedCleaner::nonRedundantSeeds(tkSeeds const& seeds) const {
  std::vector<uint> idxTriplets{}, idxNonTriplets{};
  idxTriplets.reserve(seeds.size());
  idxNonTriplets.reserve(seeds.size());

  for (uint i1 = 0; i1 < seeds.size(); ++i1) {
    auto const& s1 = seeds[i1];
    if (s1.nHits() == 3)
      idxTriplets.emplace_back(i1);
    else
      idxNonTriplets.emplace_back(i1);
  }

  if (idxTriplets.empty()) {
    return seeds;
  }

  std::vector<bool> keepSeedFlags(seeds.size(), true);
  for (uint j1 = 0; j1 < idxNonTriplets.size(); ++j1) {
    auto const i1 = idxNonTriplets[j1];
    auto const& seed = seeds[i1];
    keepSeedFlags[i1] = seedIsNotRedundant(seeds, seed, idxTriplets);
  }

  tkSeeds result{};
  result.reserve(seeds.size());

  for (uint i1 = 0; i1 < seeds.size(); ++i1) {
    if (keepSeedFlags[i1]) {
      result.emplace_back(seeds[i1]);
    }
  }

  return result;
}

bool TrackerSeedCleaner::seedIsNotRedundant(tkSeeds const& seeds,
                                            TrajectorySeed const& s1,
                                            std::vector<uint> const& otherIdxs) const {
  auto const& rh1s = s1.recHits();
  for (uint j2 = 0; j2 < otherIdxs.size(); ++j2) {
    auto const& s2 = seeds[otherIdxs[j2]];
    // number of shared hits
    uint shared = 0;
    for (auto const& h2 : s2.recHits()) {
      for (auto const& h1 : rh1s) {
        if (h2.sharesInput(&h1, TrackingRecHit::all)) {
          ++shared;
        }
      }
    }
    if (shared == 2) {
      return false;
    }
  }

  return true;
}
