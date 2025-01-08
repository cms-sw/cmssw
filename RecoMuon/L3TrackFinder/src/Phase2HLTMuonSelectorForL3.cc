/**  \class Phase2HLTMuonSelectorForL3
 *   See header file for a description of this class
 *   \author Luca Ferragina (INFN BO), 2024
 */
#include "RecoMuon/L3TrackFinder/interface/Phase2HLTMuonSelectorForL3.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <unordered_set>

// Constructor
Phase2HLTMuonSelectorForL3::Phase2HLTMuonSelectorForL3(const edm::ParameterSet& iConfig)
    : l1TkMuCollToken_(consumes<l1t::TrackerMuonCollection>(iConfig.getParameter<edm::InputTag>("l1TkMuons"))),
      l2MuCollectionToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l2MuonsUpdVtx"))),
      l3TrackCollectionToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l3Tracks"))),
      IOFirst_(iConfig.getParameter<bool>("IOFirst")),
      matchingDr_(iConfig.getParameter<double>("matchingDr")),
      applyL3Filters_(iConfig.getParameter<bool>("applyL3Filters")),
      maxNormalizedChi2_(iConfig.getParameter<double>("MaxNormalizedChi2")),
      maxPtDifference_(iConfig.getParameter<double>("MaxPtDifference")),
      minNhits_(iConfig.getParameter<int>("MinNhits")),
      minNhitsMuons_(iConfig.getParameter<int>("MinNhitsMuons")),
      minNhitsPixel_(iConfig.getParameter<int>("MinNhitsPixel")),
      minNhitsTracker_(iConfig.getParameter<int>("MinNhitsTracker")) {
  if (IOFirst_) {
    produces<reco::TrackCollection>("L2MuToReuse");
    produces<reco::TrackCollection>("L3IOTracksFiltered");
  } else {
    produces<l1t::TrackerMuonCollection>("L1TkMuToReuse");
    produces<reco::TrackCollection>("L3OITracksFiltered");
  }
}

void Phase2HLTMuonSelectorForL3::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1TkMuons", edm::InputTag("l1tTkMuonsGmt"));
  desc.add<edm::InputTag>("l2MuonsUpdVtx", edm::InputTag("hltL2MuonsFromL1TkMuon", "UpdatedAtVtx"));
  desc.add<edm::InputTag>("l3Tracks", edm::InputTag("hltIter2Phase2L3FromL1TkMuonMerged"));
  desc.add<bool>("IOFirst", true);
  desc.add<double>("matchingDr", 0.02);
  desc.add<bool>("applyL3Filters", true);
  desc.add<int>("MinNhits", 1);
  desc.add<double>("MaxNormalizedChi2", 5.0);
  desc.add<int>("MinNhitsMuons", 0);
  desc.add<int>("MinNhitsPixel", 1);
  desc.add<int>("MinNhitsTracker", 6);
  desc.add<double>("MaxPtDifference", 999.0);  //relative difference
  descriptions.add("Phase2HLTMuonSelectorForL3", desc);
}

// IO first -> collection of L2 muons not already matched to a L3 inner track
// OI first -> collection of L1Tk Muons not matched to a L3 track
void Phase2HLTMuonSelectorForL3::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "RecoMuon|Phase2HLTMuonSelectorForL3";

  // L3 tracks (IO or OI)
  auto l3TracksCollectionH = iEvent.getHandle(l3TrackCollectionToken_);

  if (IOFirst_) {
    LogDebug(metname) << "Inside-Out reconstruction done first, looping over L2 muons";

    // L2 Muons collection
    auto const l2MuonsCollectionH = iEvent.getHandle(l2MuCollectionToken_);

    // Output
    std::unique_ptr<reco::TrackCollection> L2MuToReuse = std::make_unique<reco::TrackCollection>();
    std::unique_ptr<reco::TrackCollection> L3IOTracksFiltered = std::make_unique<reco::TrackCollection>();

    // Indexes of good L3 Tracks
    std::unordered_set<size_t> goodL3Indexes;

    // Loop over L2 Muons
    for (size_t l2MuIndex = 0; l2MuIndex != l2MuonsCollectionH->size(); ++l2MuIndex) {
      reco::TrackRef l2MuRef(l2MuonsCollectionH, l2MuIndex);
      bool reuseL2 = true;

      // Extract L1TkMu from L2 Muon
      edm::RefToBase<TrajectorySeed> seedRef = l2MuRef->seedRef();
      edm::Ref<L2MuonTrajectorySeedCollection> l2Seed = seedRef.castTo<edm::Ref<L2MuonTrajectorySeedCollection>>();
      l1t::TrackerMuonRef l1TkMuRef = l2Seed->l1TkMu();

      // Check validity of cast (actually found a L1TkMu)
      if (l1TkMuRef.isNonnull()) {
        // Loop over L3 tracks
        LogDebug(metname) << "Looping over L3 tracks";
        for (size_t l3MuIndex = 0; l3MuIndex != l3TracksCollectionH->size(); ++l3MuIndex) {
          reco::TrackRef l3TrackRef(l3TracksCollectionH, l3MuIndex);
          bool rejectL3 = true;
          // Filter L3 Tracks
          if (applyL3Filters_) {
            LogDebug(metname) << "Checking L3 Track quality";
            rejectL3 = rejectL3Track(l1TkMuRef, l3TrackRef);
            if (!rejectL3) {
              LogDebug(metname) << "Adding good quality L3 IO track to filtered collection";
              goodL3Indexes.insert(l3MuIndex);
            }
          }
          // Check match in dR
          float dR2 = deltaR2(l1TkMuRef->phEta(), l1TkMuRef->phPhi(), l3TrackRef->eta(), l3TrackRef->phi());
          LogDebug(metname) << "deltaR2: " << dR2;
          if (dR2 < matchingDr_ * matchingDr_) {
            LogDebug(metname) << "Found L2 muon that matches the L3 track";
            reuseL2 = applyL3Filters_ ? rejectL3 : false;
            LogDebug(metname) << "Reuse L2: " << reuseL2;
          }
        }  // End loop over L3 Tracks
      } else {
        LogDebug(metname) << "Found L2 muon without an associated L1TkMu";
      }
      if (reuseL2) {
        LogDebug(metname) << "Found a L2 muon to be reused";
        L2MuToReuse->push_back(*l2MuRef);
      }
    }  // End loop over L2 Muons

    // Fill L3 IO Tracks Filtered
    for (const size_t index : goodL3Indexes) {
      L3IOTracksFiltered->push_back(*(reco::TrackRef(l3TracksCollectionH, index)));
    }

    LogDebug(metname) << "Placing L2 Muons to be reused in the event";
    iEvent.put(std::move(L2MuToReuse), "L2MuToReuse");
    LogDebug(metname) << "Placing good quality L3 IO Tracks in the event";
    iEvent.put(std::move(L3IOTracksFiltered), "L3IOTracksFiltered");
  } else {
    LogDebug(metname) << "Outside-In reconstruction done first, looping over L1Tk muons";

    // L1Tk Muons collection
    auto const l1TkMuonsCollectionH = iEvent.getHandle(l1TkMuCollToken_);

    // Output
    std::unique_ptr<l1t::TrackerMuonCollection> L1TkMuToReuse = std::make_unique<l1t::TrackerMuonCollection>();
    std::unique_ptr<reco::TrackCollection> L3OITracksFiltered = std::make_unique<reco::TrackCollection>();

    // Indexes of good L3 Tracks
    std::unordered_set<size_t> goodL3Indexes;

    // Loop over L1Tk Muons
    for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuonsCollectionH->size(); ++l1TkMuIndex) {
      l1t::TrackerMuonRef l1TkMuRef(l1TkMuonsCollectionH, l1TkMuIndex);
      bool reuseL1TkMu = true;

      // Loop over L3 tracks
      LogDebug(metname) << "Looping over L3 tracks";
      for (size_t l3MuIndex = 0; l3MuIndex != l3TracksCollectionH->size(); ++l3MuIndex) {
        reco::TrackRef l3TrackRef(l3TracksCollectionH, l3MuIndex);
        bool rejectL3 = true;
        // Filter L3 Tracks
        if (applyL3Filters_) {
          LogDebug(metname) << "Checking L3 Track quality";
          rejectL3 = rejectL3Track(l1TkMuRef, l3TrackRef);
          if (!rejectL3) {
            LogDebug(metname) << "Adding good quality L3 OI track to filtered collection";
            goodL3Indexes.insert(l3MuIndex);
          }
        }
        // Check match in dR
        float dR2 = deltaR2(l1TkMuRef->phEta(), l1TkMuRef->phPhi(), l3TrackRef->eta(), l3TrackRef->phi());
        LogDebug(metname) << "deltaR2: " << dR2;
        if (dR2 < matchingDr_ * matchingDr_) {
          LogDebug(metname) << "Found L1TkMu that matches the L3 track";
          reuseL1TkMu = applyL3Filters_ ? rejectL3 : false;
          LogDebug(metname) << "Reuse L1TkMu: " << reuseL1TkMu;
        }
      }  // End loop over L3 Tracks
      if (reuseL1TkMu) {
        LogDebug(metname) << "Found a L1TkMu to be reused";
        L1TkMuToReuse->push_back(*l1TkMuRef);
      }
    }  // End loop over L1Tk Muons

    // Fill L3 OI Tracks Filtered
    for (const size_t index : goodL3Indexes) {
      L3OITracksFiltered->push_back(*(reco::TrackRef(l3TracksCollectionH, index)));
    }

    LogDebug(metname) << "Placing L1Tk Muons to be reused in the event";
    iEvent.put(std::move(L1TkMuToReuse), "L1TkMuToReuse");
    LogDebug(metname) << "Placing good quality L3 OI Tracks in the event";
    iEvent.put(std::move(L3OITracksFiltered), "L3OITracksFiltered");
  }
}

const bool Phase2HLTMuonSelectorForL3::rejectL3Track(l1t::TrackerMuonRef l1TkMuRef, reco::TrackRef l3TrackRef) const {
  const std::string metname = "RecoMuon|Phase2HLTMuonSelectorForL3";

  bool nHitsCut = l3TrackRef->numberOfValidHits() < minNhits_;
  bool chi2Cut = l3TrackRef->normalizedChi2() > maxNormalizedChi2_;
  bool nHitsMuonsCut = l3TrackRef->hitPattern().numberOfValidMuonHits() < minNhitsMuons_;
  bool nHitsPixelCut = l3TrackRef->hitPattern().numberOfValidPixelHits() < minNhitsPixel_;
  bool nHitsTrackerCut = l3TrackRef->hitPattern().trackerLayersWithMeasurement() < minNhitsTracker_;
  bool ptCut = std::abs(l3TrackRef->pt() - l1TkMuRef->phPt()) > maxPtDifference_ * l3TrackRef->pt();

  bool reject = nHitsCut or chi2Cut or nHitsMuonsCut or nHitsPixelCut or nHitsTrackerCut or ptCut;

  LogDebug(metname) << "nHits: " << l3TrackRef->numberOfValidHits() << " | chi2: " << l3TrackRef->normalizedChi2()
                    << " | nHitsMuon: " << l3TrackRef->hitPattern().numberOfValidMuonHits()
                    << " | nHitsPixel: " << l3TrackRef->hitPattern().numberOfValidPixelHits()
                    << " | nHitsTracker: " << l3TrackRef->hitPattern().trackerLayersWithMeasurement();
  LogDebug(metname) << "Reject L3 Track: " << reject;
  return reject;
}
