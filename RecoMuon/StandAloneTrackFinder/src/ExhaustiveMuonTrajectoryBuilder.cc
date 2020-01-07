#include "RecoMuon/StandAloneTrackFinder/interface/ExhaustiveMuonTrajectoryBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"

ExhaustiveMuonTrajectoryBuilder::ExhaustiveMuonTrajectoryBuilder(const edm::ParameterSet& pset,
                                                                 const MuonServiceProxy* proxy,
                                                                 edm::ConsumesCollector& iC)
    : theTrajBuilder(pset, proxy, iC), theSeeder(), theService(proxy) {}

ExhaustiveMuonTrajectoryBuilder::~ExhaustiveMuonTrajectoryBuilder() {}

MuonTrajectoryBuilder::TrajectoryContainer ExhaustiveMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed) {
  LocalTrajectoryParameters localTrajectoryParameters(seed.startingState().parameters());
  LocalVector p(localTrajectoryParameters.momentum());
  int rawId = seed.startingState().detId();
  DetId detId(rawId);
  bool isBarrel = (detId.subdetId() == 1);
  // homemade local-to-global
  double pt = (isBarrel) ? -p.z() : p.perp();
  pt *= localTrajectoryParameters.charge();
  float err00 = seed.startingState().error(0);
  //   float p_err = sqr(sptmean/(ptmean*ptmean));
  //  mat[0][0]= p_err;
  float sigmapt = sqrt(err00) * pt * pt;
  TrajectorySeed::range range = seed.recHits();
  TrajectoryContainer result;
  // Make a new seed based on each segment, using the original pt and sigmapt
  for (TrajectorySeed::const_iterator recHitItr = range.first; recHitItr != range.second; ++recHitItr) {
    const GeomDet* geomDet = theService->trackingGeometry()->idToDet((*recHitItr).geographicalId());
    MuonTransientTrackingRecHit::MuonRecHitPointer muonRecHit =
        MuonTransientTrackingRecHit::specificBuild(geomDet, &*recHitItr);
    TrajectorySeed tmpSeed(theSeeder.createSeed(pt, sigmapt, muonRecHit));
    TrajectoryContainer trajectories(theTrajBuilder.trajectories(tmpSeed));
    result.insert(
        result.end(), std::make_move_iterator(trajectories.begin()), std::make_move_iterator(trajectories.end()));
  }
  return result;
}

MuonTrajectoryBuilder::CandidateContainer ExhaustiveMuonTrajectoryBuilder::trajectories(const TrackCand&) {
  return CandidateContainer();
}

void ExhaustiveMuonTrajectoryBuilder::setEvent(const edm::Event& event) { theTrajBuilder.setEvent(event); }

void ExhaustiveMuonTrajectoryBuilder::clean(TrajectoryContainer& trajectories) const {
  // choose the one with the most hits, and the smallest chi-squared
  if (trajectories.empty()) {
    return;
  }
  int best_nhits = 0;
  unsigned best = 0;
  unsigned ntraj = trajectories.size();
  for (unsigned i = 0; i < ntraj; ++i) {
    int nhits = trajectories[i]->foundHits();
    if (nhits > best_nhits) {
      best_nhits = nhits;
      best = i;
    } else if (nhits == best_nhits && trajectories[i]->chiSquared() < trajectories[best]->chiSquared()) {
      best = i;
    }
  }
  TrajectoryContainer result;
  result.reserve(1);
  result.emplace_back(std::move(trajectories[best]));
  trajectories.swap(result);
}
