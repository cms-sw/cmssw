#include "FastSimulation/Muons/plugins/FastTSGFromL2Muon.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

#include <set>

FastTSGFromL2Muon::FastTSGFromL2Muon(const edm::ParameterSet& cfg)
    : thePtCut(cfg.getParameter<double>("PtCut")),
      theL2CollectionLabel(cfg.getParameter<edm::InputTag>("MuonCollectionLabel")),
      theSeedCollectionLabels(cfg.getParameter<std::vector<edm::InputTag> >("SeedCollectionLabels")),
      theSimTrackCollectionLabel(cfg.getParameter<edm::InputTag>("SimTrackCollectionLabel")),
      simTrackToken_(consumes<edm::SimTrackContainer>(theSimTrackCollectionLabel)),
      l2TrackToken_(consumes<reco::TrackCollection>(theL2CollectionLabel)) {
  produces<L3MuonTrajectorySeedCollection>();

  for (auto& seed : theSeedCollectionLabels)
    seedToken_.emplace_back(consumes<edm::View<TrajectorySeed> >(seed));

  edm::ParameterSet regionBuilderPSet = cfg.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");
  theRegionBuilder = std::make_unique<MuonTrackingRegionBuilder>(regionBuilderPSet, consumesCollector());
}

void FastTSGFromL2Muon::beginRun(edm::Run const& run, edm::EventSetup const& es) {
  //region builder
}

void FastTSGFromL2Muon::produce(edm::Event& ev, const edm::EventSetup& es) {
  // Initialize the output product
  std::unique_ptr<L3MuonTrajectorySeedCollection> result(new L3MuonTrajectorySeedCollection());

  // Region builder
  theRegionBuilder->setEvent(ev, es);

  // Retrieve the Monte Carlo truth (SimTracks)
  const edm::Handle<edm::SimTrackContainer>& theSimTracks = ev.getHandle(simTrackToken_);

  // Retrieve L2 muon collection
  const edm::Handle<reco::TrackCollection>& l2muonH = ev.getHandle(l2TrackToken_);

  // Retrieve Seed collection
  unsigned seedCollections = theSeedCollectionLabels.size();
  std::vector<edm::Handle<edm::View<TrajectorySeed> > > theSeeds;
  theSeeds.resize(seedCollections);
  unsigned seed_size = 0;
  for (unsigned iseed = 0; iseed < seedCollections; ++iseed) {
    ev.getByToken(seedToken_[iseed], theSeeds[iseed]);
    seed_size += theSeeds[iseed]->size();
  }

  // Loop on L2 muons
  unsigned int imu = 0;
  unsigned int imuMax = l2muonH->size();
  edm::LogVerbatim("FastTSGFromL2Muon") << "Found " << imuMax << " L2 muons";
  for (; imu != imuMax; ++imu) {
    // Make a ref to l2 muon
    reco::TrackRef muRef(l2muonH, imu);

    // Cut on muons with low momenta
    if (muRef->pt() < thePtCut || muRef->innerMomentum().Rho() < thePtCut || muRef->innerMomentum().R() < 2.5)
      continue;

    // Define the region of interest
    std::unique_ptr<RectangularEtaPhiTrackingRegion> region = theRegionBuilder->region(muRef);

    // Copy the collection of seeds (ahem, this is time consuming!)
    std::vector<TrajectorySeed> tkSeeds;
    std::set<unsigned> tkIds;
    tkSeeds.reserve(seed_size);
    for (unsigned iseed = 0; iseed < seedCollections; ++iseed) {
      edm::Handle<edm::View<TrajectorySeed> > aSeedCollection = theSeeds[iseed];
      unsigned nSeeds = aSeedCollection->size();
      for (unsigned seednr = 0; seednr < nSeeds; ++seednr) {
        // The seed
        const BasicTrajectorySeed* aSeed = &((*aSeedCollection)[seednr]);

        // The SimTrack id associated to the first hit of the Seed
        int simTrackId = static_cast<FastTrackerRecHit const&>(*aSeed->recHits().begin()).simTrackId(0);

        // Track already associated to a seed
        std::set<unsigned>::iterator tkId = tkIds.find(simTrackId);
        if (tkId != tkIds.end())
          continue;

        const SimTrack& theSimTrack = (*theSimTracks)[simTrackId];

        if (clean(muRef, region.get(), aSeed, theSimTrack))
          tkSeeds.push_back(*aSeed);
        tkIds.insert(simTrackId);

      }  // End loop on seeds

    }  // End loop on seed collections

    // Now create the Muon Trajectory Seed
    unsigned int is = 0;
    unsigned int isMax = tkSeeds.size();
    for (; is != isMax; ++is) {
      result->push_back(L3MuonTrajectorySeed(tkSeeds[is], muRef));
    }  // End of tk seed loop

  }  // End of l2 muon loop

  edm::LogVerbatim("FastTSGFromL2Muon") << "Found " << result->size() << " seeds for muons";

  //put in the event
  ev.put(std::move(result));
}

bool FastTSGFromL2Muon::clean(reco::TrackRef muRef,
                              RectangularEtaPhiTrackingRegion* region,
                              const BasicTrajectorySeed* aSeed,
                              const SimTrack& theSimTrack) {
  // Eta cleaner
  const PixelRecoRange<float>& etaRange = region->etaRange();
  double etaSeed = theSimTrack.momentum().Eta();
  double etaLimit = (fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) < 0.05)
                        ? 0.05
                        : fabs(fabs(etaRange.max()) - fabs(etaRange.mean()));
  bool inEtaRange = etaSeed >= (etaRange.mean() - etaLimit) && etaSeed <= (etaRange.mean() + etaLimit);
  if (!inEtaRange)
    return false;

  // Phi cleaner
  const TkTrackingRegionsMargin<float>& phiMargin = region->phiMargin();
  double phiSeed = theSimTrack.momentum().Phi();
  double phiLimit = (phiMargin.right() < 0.05) ? 0.05 : phiMargin.right();
  bool inPhiRange = (fabs(deltaPhi(phiSeed, double(region->direction().phi()))) < phiLimit);
  if (!inPhiRange)
    return false;

  // pt cleaner
  double ptSeed = std::sqrt(theSimTrack.momentum().Perp2());
  double ptMin = (region->ptMin() > 3.5) ? 3.5 : region->ptMin();
  bool inPtRange = ptSeed >= ptMin && ptSeed <= 2 * (muRef->pt());
  return inPtRange;
}
