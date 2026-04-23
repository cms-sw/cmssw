/**  \class MuonSeedsSelectorFromL1TkMuon
 *
 *  \brief Selection of TrajectorySeeds that match L1 tracker muons
 *
 *  The module selects the TrajectorySeeds that match an L1 tracker muon (l1t::TrackerMuon).
 *  The arbitration is similar to the one of MuonIdProducer:
 *
 *  1. A loose pre-selection in a dR cone around the L1 tracker muon.
 *  2. A quality from the seed properties (number of hits) and the pT compatibility.
 *  3. Optionally, a hit-by-hit check of the OT hits against the stubs of the L1 track.
 *  4. The arbitration: for each L1 tracker muon, keep the N seeds with the best combined quality.
 *
 *  The module keeps all seeds in the cone with pT above maxPtForCompatibilityCheck, with no other
 *  requirement. It arbitrates the seeds below this pT for each L1 tracker muon, for all L1 tracker
 *  muon pT values, because the measured pT of a genuine seed can be below the threshold.
 *
 *  \author Luca Ferragina (INFN BO)
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"

#include <set>
#include <unordered_set>
#include <vector>
#include <algorithm>

class MuonSeedsSelectorFromL1TkMuon : public edm::stream::EDProducer<> {
public:
  MuonSeedsSelectorFromL1TkMuon(const edm::ParameterSet& iConfig);
  ~MuonSeedsSelectorFromL1TkMuon() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  struct SeedInfo {
    // DetIds of the OT stacks crossed by the seed hits, used for consistency check with L1TkMu stubs
    std::vector<DetId> otStackIds;

    // Index of the seed in the input collection
    size_t seedIndex;

    // Kinematics
    float eta;
    float phi;
    float pt;

    // Hits
    int nHits;
    int nPixelHits;
    int nOTHits;
    int nConsistentHits;

    // Flag seeds to be skipped based on basic criteria (e.g. invalid TSOS)
    bool skip = false;
  };

  struct SeedQuality {
    // Index of the seed in the input collection
    size_t seedIndex;

    // Kinematics matching
    float dR2;
    float normalizedPtDiff;

    // Hits
    int nHits;
    int nPixelHits;
    int nOTHits;
    int nConsistentHits;

    // Quality terms
    float termDr;
    float termConsistency;
    float termNHits;
    float termPtDiff;
    float quality;

    bool operator<(const SeedQuality& other) const { return quality < other.quality; }
  };

  // Cache seed information to avoid redundant calculations during quality evaluation
  SeedInfo fillSeedInfo(const TrajectorySeed& seed,
                        const size_t seedIndex,
                        const GlobalTrackingGeometry* geometry,
                        const MagneticField* magField,
                        const TrackerTopology* tkTopo,
                        const reco::BeamSpot& beamSpot) const;

  // Check how many OT hits in a seed are consistent with L1TkMu stubs
  int countConsistentHits(const std::vector<DetId>& hitIds, const std::unordered_set<DetId>& stubDetIds) const;

  // Compute seed quality based on kinematic matching and hit consistency with L1TkMu stubs
  SeedQuality evaluateSeedQuality(const SeedInfo& sinfo,
                                  const float deltaR2,
                                  const float normPtDiff,
                                  const std::unordered_set<DetId>& stubDetIds) const;

  // Tokens
  const edm::EDGetTokenT<l1t::TrackerMuonCollection> l1TkMuToken_;
  const edm::EDGetTokenT<TrajectorySeedCollection> seedToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geometryToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;

  // Selection parameters
  const float l1TkMuMinPt_;
  const float maxDrForPreselection_;
  const float pTCompatibility_;
  const float maxPtForCompatibilityCheck_;

  // Arbitration parameters
  const int minNPixelHits_;
  const int minNConsistentHits_;
  const float maxCombinedQuality_;
  const size_t nSeedsToKeep_;
  const bool doConsistencyCheckWithL1Stubs_;
};

MuonSeedsSelectorFromL1TkMuon::MuonSeedsSelectorFromL1TkMuon(const edm::ParameterSet& iConfig)
    : l1TkMuToken_{
          consumes<l1t::TrackerMuonCollection>(iConfig.getParameter<edm::InputTag>("L1TkMuonInputCollection"))},
      seedToken_{consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("SeedInputCollection"))},
      beamSpotToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      magFieldToken_{esConsumes<MagneticField, IdealMagneticFieldRecord>()},
      geometryToken_{esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()},
      tkTopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      l1TkMuMinPt_{static_cast<float>(iConfig.getParameter<double>("L1TkMuMinPt"))},
      maxDrForPreselection_{static_cast<float>(iConfig.getParameter<double>("maxDrForPreselection"))},
      pTCompatibility_{static_cast<float>(iConfig.getParameter<double>("pTCompatibility"))},
      maxPtForCompatibilityCheck_{static_cast<float>(iConfig.getParameter<double>("maxPtForCompatibilityCheck"))},
      minNPixelHits_{iConfig.getParameter<int>("minNPixelHits")},
      minNConsistentHits_{iConfig.getParameter<int>("minNConsistentHits")},
      maxCombinedQuality_{static_cast<float>(iConfig.getParameter<double>("maxCombinedQuality"))},
      nSeedsToKeep_{static_cast<size_t>(iConfig.getParameter<uint>("nSeedsToKeep"))},
      doConsistencyCheckWithL1Stubs_{iConfig.getParameter<bool>("doConsistencyCheckWithL1Stubs")} {
  produces<TrajectorySeedCollection>();
}

MuonSeedsSelectorFromL1TkMuon::SeedInfo MuonSeedsSelectorFromL1TkMuon::fillSeedInfo(
    const TrajectorySeed& seed,
    const size_t seedIndex,
    const GlobalTrackingGeometry* geometry,
    const MagneticField* magField,
    const TrackerTopology* tkTopo,
    const reco::BeamSpot& beamSpot) const {
  SeedInfo info;
  info.seedIndex = seedIndex;
  info.nHits = seed.nHits();
  info.otStackIds.reserve(seed.nHits());

  // Check if seed has valid starting state and can be converted to global kinematics
  const GeomDet* seedDet = geometry->idToDet(seed.startingState().detId());
  if (!seedDet) {
    info.skip = true;
    return info;
  }

  TrajectoryStateOnSurface seedTSOS =
      trajectoryStateTransform::transientState(seed.startingState(), &(seedDet->surface()), magField);
  if (!seedTSOS.isValid()) {
    info.skip = true;
    return info;
  }

  // Use the seed direction at the point of closest approach to the beam line, because the module compares it with
  // the direction of the L1 track at the vertex. The starting state is on the outermost seed hit, at R up to about
  // 1 m for LST seeds. There, the field has turned the direction of a low-pT track by about 0.3 B R / pT (about
  // 0.5 rad at 2.3 GeV).
  const TrajectoryStateClosestToBeamLine atBeamLine = TSCBLBuilderNoMaterial()(*seedTSOS.freeState(), beamSpot);
  const GlobalVector momentum =
      atBeamLine.isValid() ? atBeamLine.trackStateAtPCA().momentum() : seedTSOS.globalMomentum();
  info.eta = momentum.eta();
  info.phi = momentum.phi();
  info.pt = momentum.perp();

  info.nPixelHits = 0;
  info.nOTHits = 0;
  info.nConsistentHits = -1;

  for (auto const& recHit : seed.recHits()) {
    if (!recHit.isValid()) {
      continue;
    }

    DetId hitId = recHit.geographicalId();

    if (hitId.subdetId() == PixelSubdetector::PixelBarrel || hitId.subdetId() == PixelSubdetector::PixelEndcap) {
      info.nPixelHits++;
    } else if (hitId.subdetId() == StripSubdetector::TOB || hitId.subdetId() == StripSubdetector::TID) {
      info.nOTHits++;
      // The DetId of a stub is the stack (PS or 2S module) of the stub. The DetId of a rec-hit is one sensor of
      // the stack. Thus, convert the rec-hit DetId to its stack DetId to compare it with the stubs.
      if (doConsistencyCheckWithL1Stubs_) {
        info.otStackIds.push_back(tkTopo->stack(hitId));
      }
    } else {
      edm::LogWarning("MuonSeedsSelectorFromL1TkMuon")
          << "Seed has hit with unexpected subdetector ID " << hitId.subdetId() << " (det " << hitId.det() << ")";
    }
  }
  if (info.nHits != info.nPixelHits + info.nOTHits) {
    edm::LogWarning("MuonSeedsSelectorFromL1TkMuon")
        << "Seed hit counting inconsistency: total nHits=" << info.nHits << " but pixelHits=" << info.nPixelHits
        << " and otHits=" << info.nOTHits;
  }

  return info;
}

int MuonSeedsSelectorFromL1TkMuon::countConsistentHits(const std::vector<DetId>& hitIds,
                                                       const std::unordered_set<DetId>& stubDetIds) const {
  int nConsistent = 0;
  for (const auto& hitId : hitIds) {
    if (stubDetIds.count(hitId) > 0) {
      nConsistent++;
    }
  }

  return nConsistent;
}

MuonSeedsSelectorFromL1TkMuon::SeedQuality MuonSeedsSelectorFromL1TkMuon::evaluateSeedQuality(
    const SeedInfo& sinfo,
    const float deltaR2,
    const float normPtDiff,
    const std::unordered_set<DetId>& stubDetIds) const {
  const std::string metname = "RecoMuon|L3TrackFinder|MuonSeedsSelectorFromL1TkMuon";

  SeedQuality squal;
  squal.seedIndex = sinfo.seedIndex;

  squal.nHits = sinfo.nHits;
  squal.nPixelHits = sinfo.nPixelHits;
  squal.nOTHits = sinfo.nOTHits;
  squal.nConsistentHits = doConsistencyCheckWithL1Stubs_ ? countConsistentHits(sinfo.otStackIds, stubDetIds) : 0;

  squal.dR2 = deltaR2;
  squal.normalizedPtDiff = normPtDiff;

  // Quality metric: lower is better, 0 is perfect.
  // Term 1: dR^2 divided by the maximum dR^2 of the pre-selection
  float termDr = squal.dR2 / (maxDrForPreselection_ * maxDrForPreselection_);
  squal.termDr = termDr;

  // Term 2: fraction of the OT hits that agree with the stubs of the L1 track
  float termConsistency = squal.nOTHits > 0 && doConsistencyCheckWithL1Stubs_
                              ? static_cast<float>(squal.nConsistentHits) / squal.nOTHits
                              : 0.0f;
  squal.termConsistency = termConsistency;

  // Term 3: inverse of the number of hits
  float termNHits = squal.nHits > 0 ? 1.0f / squal.nHits : 1e6;
  squal.termNHits = termNHits;

  // Term 4: relative pT difference, divided by its maximum
  float termPtDiff = squal.normalizedPtDiff / pTCompatibility_;
  squal.termPtDiff = termPtDiff;

  // Combined quality: the sum of the terms for seeds with pT up to maxPtForCompatibilityCheck, else only term 1
  squal.quality = sinfo.pt <= maxPtForCompatibilityCheck_ ? termDr - termConsistency + termNHits + termPtDiff : termDr;

  LogDebug(metname) << "Seed " << squal.seedIndex << " quality evaluation:"
                    << "\n  nHits=" << squal.nHits << " nPixelHits=" << squal.nPixelHits
                    << " nConsistentHits=" << squal.nConsistentHits << " dR2=" << squal.dR2
                    << "\n  combined quality=" << squal.quality;

  return squal;
}

void MuonSeedsSelectorFromL1TkMuon::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "RecoMuon|L3TrackFinder|MuonSeedsSelectorFromL1TkMuon";

  // Output collection
  std::unique_ptr<TrajectorySeedCollection> outputSeeds = std::make_unique<TrajectorySeedCollection>();

  // Get input collections
  auto const l1TkMuCollectionH = iEvent.getHandle(l1TkMuToken_);
  auto const seedCollectionH = iEvent.getHandle(seedToken_);

  if (!l1TkMuCollectionH.isValid() || !seedCollectionH.isValid()) {
    edm::LogWarning(metname) << "Input collections not valid! Returning empty collection.";
    iEvent.put(std::move(outputSeeds));
    return;
  }

  if (l1TkMuCollectionH->empty() || seedCollectionH->empty()) {
    LogDebug(metname) << "Input collections empty! Returning empty collection.";
    iEvent.put(std::move(outputSeeds));
    return;
  }

  // Get event setup
  const MagneticField* magField = &iSetup.getData(magFieldToken_);
  const GlobalTrackingGeometry* geometry = &iSetup.getData(geometryToken_);
  const TrackerTopology* tkTopo = &iSetup.getData(tkTopoToken_);
  const reco::BeamSpot& beamSpot = iEvent.get(beamSpotToken_);

  // Store indices of matched seeds
  std::set<size_t> seedsToKeep;

  LogDebug(metname) << "Processing " << l1TkMuCollectionH->size() << " L1TkMuons and " << seedCollectionH->size()
                    << " seeds";

  // Cache seeds info
  std::vector<SeedInfo> seedsInfo;
  seedsInfo.reserve(seedCollectionH->size());
  for (size_t seedIndex = 0; seedIndex != seedCollectionH->size(); ++seedIndex) {
    SeedInfo info = fillSeedInfo((*seedCollectionH)[seedIndex], seedIndex, geometry, magField, tkTopo, beamSpot);
    if (!info.skip) {
      seedsInfo.push_back(info);
    } else {
      LogDebug(metname) << "Skipping seed " << seedIndex << " due to invalid TSOS or geometry";
    }
  }

  // Loop over L1TkMuons
  for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuCollectionH->size(); ++l1TkMuIndex) {
    l1t::TrackerMuonRef l1TkMuRef(l1TkMuCollectionH, l1TkMuIndex);

    // Kinematics of the L1 track, as stored in the L1 tracker muon. They are available also when the L1 track
    // collection is not in the event.
    const float l1TkMuEta = l1TkMuRef->eta();
    const float l1TkMuPhi = l1TkMuRef->phi();
    const float l1TkMuPt = l1TkMuRef->pt();

    // Skip L1TkMu below pT threshold
    if (l1TkMuPt < l1TkMuMinPt_) {
      LogDebug(metname) << "Skipping L1TkMuon " << l1TkMuIndex << " with pT=" << l1TkMuPt << " below threshold of "
                        << l1TkMuMinPt_;
      continue;
    }
    LogDebug(metname) << "Processing L1TkMuon " << l1TkMuIndex << " with eta=" << l1TkMuEta << ", phi=" << l1TkMuPhi
                      << ", pT=" << l1TkMuPt;

    // Detector ids of the stubs of the L1 track, for the consistency check. Without the L1 track in the event the
    // set stays empty: no seed hit is consistent, and the selection is the same as without the check.
    std::unordered_set<DetId> stubDetIds;
    if (doConsistencyCheckWithL1Stubs_) {
      const auto& trkPtr = l1TkMuRef->trkPtr();
      if (trkPtr.isAvailable()) {
        for (const auto& stubRef : trkPtr->getStubRefs()) {
          auto stubDetId = stubRef->getDetId();
          stubDetIds.insert(stubDetId);
          LogDebug(metname) << "  Stub detId: " << stubDetId.rawId() << " (subdet " << stubDetId.subdetId() << ")";
        }
      } else {
        LogDebug(metname) << "L1 track of L1TkMuon " << l1TkMuIndex << " not available, no stub consistency check";
      }
    }

    // Vector to hold seed quality information for this L1TkMu
    std::vector<SeedQuality> candidateSeedsWithPriority;
    std::vector<SeedQuality> candidateSeeds;

    // Loop over the cached seed information
    for (const auto& sinfo : seedsInfo) {
      // Loose dR cone for pre-selection
      float dR2 = deltaR2(l1TkMuEta, l1TkMuPhi, sinfo.eta, sinfo.phi);

      if (dR2 > maxDrForPreselection_ * maxDrForPreselection_) {
        continue;
      }
      if (sinfo.pt > maxPtForCompatibilityCheck_) {
        LogDebug(metname)
            << "Seed " << sinfo.seedIndex << " geometrically compatible with L1TkMu " << l1TkMuIndex
            << " with pT above threshold for compatibility check, skipping all other compatibility requirements";
        seedsToKeep.insert(sinfo.seedIndex);
        continue;
      }

      float normPtDiff = std::abs(sinfo.pt - l1TkMuPt) / l1TkMuPt;

      LogDebug(metname) << "Processing seed " << sinfo.seedIndex << ": eta=" << sinfo.eta << " phi=" << sinfo.phi
                        << " pT=" << sinfo.pt << " dR2 to L1TkMu " << l1TkMuIndex << " is " << dR2
                        << ", normalised pT difference=" << normPtDiff;

      if (normPtDiff > pTCompatibility_) {
        LogDebug(metname) << "Seed " << sinfo.seedIndex
                          << " rejected: pT difference with L1TkMu is too large (|seedPt - l1TkMuPt|/l1TkMuPt = "
                          << normPtDiff << " > " << pTCompatibility_;
        continue;
      }

      LogDebug(metname) << "Seed " << sinfo.seedIndex << " passed pre-selection for L1TkMu " << l1TkMuIndex
                        << " with dR2=" << dR2 << " and normalized pT difference=" << normPtDiff;

      // Evaluate seed quality
      SeedQuality squal = evaluateSeedQuality(sinfo, dR2, normPtDiff, stubDetIds);

      if (squal.quality > maxCombinedQuality_) {
        LogDebug(metname) << "Seed " << sinfo.seedIndex << " rejected: poor combined quality (" << squal.quality
                          << " > " << maxCombinedQuality_ << ")";
        continue;
      }

      // The seed passed the selection: add it to a candidate list for the arbitration. Seeds with enough pixel hits
      // and enough hits that agree with the stubs of the L1 track get priority.
      if (squal.nPixelHits >= minNPixelHits_ &&
          (squal.nConsistentHits >= minNConsistentHits_ && doConsistencyCheckWithL1Stubs_)) {
        candidateSeedsWithPriority.push_back(squal);
      } else {
        candidateSeeds.push_back(squal);
      }
      LogDebug(metname) << "Seed " << sinfo.seedIndex << " passed selection for L1TkMu " << l1TkMuIndex
                        << " with quality " << squal.quality << " (nHits=" << squal.nHits
                        << ", nPixelHits=" << squal.nPixelHits << ", nConsistentHits=" << squal.nConsistentHits
                        << ", dR2=" << squal.dR2 << ", normPtDiff=" << squal.normalizedPtDiff << ")";
    }  // End loop on seeds

    // Arbitrate the seeds with pT up to maxPtForCompatibilityCheck_ (the module kept the other seeds above). Do this
    // for each L1TkMuon, also for high-pT L1TkMuons, because the measured pT of their genuine seeds can be below the
    // threshold.
    {
      // Take the best priority seeds first, then fill up to nSeedsToKeep_ with the other candidates
      std::vector<SeedQuality> finalSeeds;
      finalSeeds.reserve(nSeedsToKeep_);
      if (!candidateSeedsWithPriority.empty()) {
        // Sort only the best nSeedsToKeep_ candidates
        size_t nToSort = std::min(nSeedsToKeep_, candidateSeedsWithPriority.size());
        std::partial_sort(candidateSeedsWithPriority.begin(),
                          candidateSeedsWithPriority.begin() + nToSort,
                          candidateSeedsWithPriority.end());

        LogDebug(metname) << "L1TkMu " << l1TkMuIndex << ": found " << candidateSeedsWithPriority.size()
                          << " candidate seeds with priority, keeping best " << nToSort;

        for (size_t i = 0; i < nToSort; ++i) {
          finalSeeds.push_back(candidateSeedsWithPriority[i]);
          auto candidate = candidateSeedsWithPriority[i];

          LogDebug(metname) << "  Keeping priority seed " << candidate.seedIndex
                            << " with quality=" << candidate.quality << " (nHits=" << candidate.nHits
                            << ", nPixel=" << candidate.nPixelHits << ", nConsistent=" << candidate.nConsistentHits
                            << ", dR2=" << candidate.dR2 << ", normPtDiff=" << candidate.normalizedPtDiff << ")";
        }
      }
      if (finalSeeds.size() < nSeedsToKeep_ and not candidateSeeds.empty()) {
        size_t nRemaining = nSeedsToKeep_ - finalSeeds.size();
        size_t nToSort = std::min(nRemaining, candidateSeeds.size());
        std::partial_sort(candidateSeeds.begin(), candidateSeeds.begin() + nToSort, candidateSeeds.end());

        LogDebug(metname) << "L1TkMu " << l1TkMuIndex << ": found " << candidateSeeds.size()
                          << " additional candidate seeds, keeping best " << nToSort;

        for (size_t i = 0; i < nToSort; ++i) {
          finalSeeds.push_back(candidateSeeds[i]);
          auto candidate = candidateSeeds[i];

          LogDebug(metname) << "  Keeping additional seed " << candidate.seedIndex
                            << " with quality=" << candidate.quality << " (nHits=" << candidate.nHits
                            << ", nPixel=" << candidate.nPixelHits << ", nConsistent=" << candidate.nConsistentHits
                            << ", dR2=" << candidate.dR2 << ", normPtDiff=" << candidate.normalizedPtDiff << ")";
        }
      }
      if (finalSeeds.empty()) {
        LogDebug(metname) << "L1TkMu " << l1TkMuIndex << ": no candidate seed passed selection";
      }
      // Add selected seeds to output
      for (const auto& seed : finalSeeds) {
        seedsToKeep.insert(seed.seedIndex);

        LogDebug(metname) << "L1TkMu " << l1TkMuIndex << " keeping seed " << seed.seedIndex
                          << " with quality=" << seed.quality << " (nHits=" << seed.nHits
                          << ", nPixel=" << seed.nPixelHits << ", nConsistent=" << seed.nConsistentHits
                          << ", dR2=" << seed.dR2 << ", normPtDiff=" << seed.normalizedPtDiff << ")";
      }
    }
  }  // End L1TkMuon loop

  LogDebug(metname) << "Seed selection from " << l1TkMuCollectionH->size() << " L1TkMuons and "
                    << seedCollectionH->size() << " input seeds: selected " << seedsToKeep.size() << " seeds";

  // Fill output collection with selected seeds
  outputSeeds->reserve(seedsToKeep.size());
  for (size_t idx : seedsToKeep) {
    outputSeeds->push_back((*seedCollectionH)[idx]);
  }
  iEvent.put(std::move(outputSeeds));
}

void MuonSeedsSelectorFromL1TkMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Input collections
  desc.add<edm::InputTag>("L1TkMuonInputCollection", edm::InputTag("l1tTkMuonsGmt"))->setComment("L1TkMuons to match");
  desc.add<edm::InputTag>("SeedInputCollection", edm::InputTag("hltInitialStepTrajectorySeedsLST"))
      ->setComment("Input seeds");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"))
      ->setComment(
          "Beam spot. The module compares the seed directions with the L1TkMuons at the point of closest approach "
          "to the beam line.");

  // L1TkMuon selection
  desc.add<double>("L1TkMuMinPt", 0.0)->setComment("Minimum pT of the L1TkMuons to match");

  // Matching parameters
  desc.add<double>("maxDrForPreselection", 0.4)->setComment("Maximum dR between the seed and the L1TkMuon");
  desc.add<double>("pTCompatibility", 0.5)
      ->setComment("Maximum relative pT difference between the seed and the L1TkMuon (|seedPt - l1TkMuPt|/l1TkMuPt)");
  desc.add<double>("maxPtForCompatibilityCheck", 25.0)
      ->setComment(
          "Seeds with pT above this value pass with only the dR selection. Seeds below it get the full "
          "compatibility check and the arbitration.");

  // Quality
  desc.add<double>("maxCombinedQuality", 2.0)->setComment("Maximum combined quality of a seed (lower is better)");

  // Arbitration
  desc.add<bool>("doConsistencyCheckWithL1Stubs", false)
      ->setComment("Compare the OT hits of the seed with the stubs of the L1 track, hit by hit");
  desc.add<int>("minNPixelHits", 2)
      ->setComment(
          "Minimum number of pixel hits for a seed to get priority in the arbitration (used only with "
          "doConsistencyCheckWithL1Stubs)");
  desc.add<int>("minNConsistentHits", 4)
      ->setComment(
          "Minimum number of OT hits that agree with the stubs of the L1 track for a seed to get priority in the "
          "arbitration (used only with doConsistencyCheckWithL1Stubs)");
  desc.add<uint32_t>("nSeedsToKeep", 2)
      ->setComment(
          "Maximum number of seeds to keep for each L1TkMuon, from the quality ranking. It does not apply to seeds "
          "with pT above maxPtForCompatibilityCheck: the module keeps all of them in the dR cone.");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonSeedsSelectorFromL1TkMuon);
