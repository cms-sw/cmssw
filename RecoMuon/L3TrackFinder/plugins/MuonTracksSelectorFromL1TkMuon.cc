/**  \class MuonTracksSelectorFromL1TkMuon
 *
 *  \brief Selection of reco::Tracks that match L1 tracker muons
 *
 *  The module selects the reco::Tracks that match an L1 tracker muon (l1t::TrackerMuon).
 *  After a loose pre-selection in dZ and dR, the module ranks the tracks by a combined quality
 *  from dR, dZ and the curvature compatibility with the L1 tracker muon. For each L1 tracker
 *  muon, it keeps up to nTracksToKeep tracks with the best quality.
 *
 *  \author Luca Ferragina (INFN BO)
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1TMuonPhase2/interface/Constants.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <set>

namespace {
  // The GMT keeps the z0 of the L1 track word with its lowest bits removed. Thus, one unit of hwZ0() is
  // 2^(BITSTTZ0 - BITSZ0) steps of the track word. phZ0() uses a different unit.
  constexpr double kL1TkMuZ0Unit = (1 << (Phase2L1GMT::BITSTTZ0 - Phase2L1GMT::BITSZ0)) * TTTrack_TrackWord::stepZ0;
}  // namespace

class MuonTracksSelectorFromL1TkMuon : public edm::stream::EDProducer<> {
public:
  MuonTracksSelectorFromL1TkMuon(const edm::ParameterSet& iConfig);

  ~MuonTracksSelectorFromL1TkMuon() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // Tokens
  const edm::EDGetTokenT<l1t::TrackerMuonCollection> l1TkMuToken_;
  const edm::EDGetTokenT<reco::TrackCollection> tkToken_;
  // Track selection parameters
  const float l1TkMuMinPt_;
  const float tkMaxPtForMatch_;
  const float maxDz_;
  const float maxDr_;
  const float maxChi2_;
  const float maxCombinedQuality_;
  const size_t nTracksToKeep_;
};

MuonTracksSelectorFromL1TkMuon::MuonTracksSelectorFromL1TkMuon(const edm::ParameterSet& iConfig)
    : l1TkMuToken_{
          consumes<l1t::TrackerMuonCollection>(iConfig.getParameter<edm::InputTag>("L1TkMuonInputCollection"))},
      tkToken_{consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("TrackInputCollection"))},
      l1TkMuMinPt_{static_cast<float>(iConfig.getParameter<double>("L1TkMuMinPt"))},
      tkMaxPtForMatch_{static_cast<float>(iConfig.getParameter<double>("trackMaxPtForMatch"))},
      maxDz_{static_cast<float>(iConfig.getParameter<double>("maxDz"))},
      maxDr_{static_cast<float>(iConfig.getParameter<double>("maxDr"))},
      maxChi2_{static_cast<float>(iConfig.getParameter<double>("maxChi2"))},
      maxCombinedQuality_{static_cast<float>(iConfig.getParameter<double>("maxCombinedQuality"))},
      nTracksToKeep_{static_cast<size_t>(iConfig.getParameter<uint>("nTracksToKeep"))} {
  produces<reco::TrackCollection>();
}

void MuonTracksSelectorFromL1TkMuon::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "RecoMuon|L3TrackFinder|MuonTracksSelectorFromL1TkMuon";

  // Output collection
  std::unique_ptr<reco::TrackCollection> outputTracks = std::make_unique<reco::TrackCollection>();

  // Get input collections
  auto const l1TkMuCollectionH = iEvent.getHandle(l1TkMuToken_);
  auto const tkCollectionH = iEvent.getHandle(tkToken_);
  if (!l1TkMuCollectionH.isValid() || !tkCollectionH.isValid() || l1TkMuCollectionH->empty() ||
      tkCollectionH->empty()) {
    LogDebug(metname) << "Input collection not valid or empty! Returning empty collection." << '\n';
    iEvent.put(std::move(outputTracks));
    return;
  }

  // Store indices of matched tracks
  std::set<size_t> tracksToKeep;

  // Loop over L1TkMuons
  for (size_t l1TkMuIndex = 0; l1TkMuIndex != l1TkMuCollectionH->size(); ++l1TkMuIndex) {
    l1t::TrackerMuonRef l1TkMuRef(l1TkMuCollectionH, l1TkMuIndex);
    // Kinematics of the L1 track, as stored in the L1 tracker muon. They are available also when the L1 track
    // collection is not in the event.
    const float l1TkMuEta = l1TkMuRef->eta();
    const float l1TkMuPhi = l1TkMuRef->phi();
    const float l1TkMuPt = l1TkMuRef->pt();
    const float l1TkMuZ0 = l1TkMuRef->hwZ0() * kL1TkMuZ0Unit;

    // Basic kinematic selection
    if (l1TkMuPt < l1TkMuMinPt_) {
      continue;
    }

    // Quality and track index of each candidate match
    std::vector<std::pair<float, size_t>> potentialMatches;

    for (size_t tkIndex = 0; tkIndex != tkCollectionH->size(); ++tkIndex) {
      reco::TrackRef tkRef(tkCollectionH, tkIndex);

      // Loose pre-selection
      float dZ = std::abs(tkRef->dz() - l1TkMuZ0);
      if (dZ > 5.0) {
        continue;
      }

      float dR2 = deltaR2(l1TkMuEta, l1TkMuPhi, tkRef->eta(), tkRef->phi());
      if (dR2 > 0.4 * 0.4) {
        continue;
      }

      // Curvature compatibility
      float l1TkMuQoverPt = float(l1TkMuRef->phCharge()) / l1TkMuPt;
      float tkQoverPt = float(tkRef->charge()) / tkRef->pt();
      float tkQoverPtError = tkRef->ptError() / (tkRef->pt() * tkRef->pt());

      float curvDiff = l1TkMuQoverPt - tkQoverPt;
      float chi2Curv = (curvDiff * curvDiff) / (tkQoverPtError * tkQoverPtError);

      // Combine quality metrics
      float termDr = dR2 / (maxDr_ * maxDr_);        // (dR / maxDr)^2
      float termDz = (dZ * dZ) / (maxDz_ * maxDz_);  // (dZ / maxDz)^2
      float termChi2 = chi2Curv / maxChi2_;          // chi2 / maxChi2

      float combinedQuality = tkRef->pt() < tkMaxPtForMatch_ ? termDr + termDz + termChi2 : termDr + termDz;

      if (combinedQuality < maxCombinedQuality_) {
        potentialMatches.emplace_back(combinedQuality, tkIndex);
      }
    }  // End Track Loop

    // Sort by quality (lowest is best)
    std::partial_sort(potentialMatches.begin(),
                      potentialMatches.begin() + std::min(nTracksToKeep_, potentialMatches.size()),
                      potentialMatches.end());

    // Keep at most nTracksToKeep_ best matches per L1TkMuon
    for (size_t i = 0; i < std::min(potentialMatches.size(), nTracksToKeep_); ++i) {
      reco::TrackRef tkRef(tkCollectionH, potentialMatches[i].second);
      tracksToKeep.insert(potentialMatches[i].second);

      LogDebug(metname) << "L1Tk muon with index " << l1TkMuIndex << " matched to track with index "
                        << potentialMatches[i].second << " with combinedQuality = " << potentialMatches[i].first << '\n'
                        << "    -> dR = " << std::sqrt(deltaR2(l1TkMuEta, l1TkMuPhi, tkRef->eta(), tkRef->phi()))
                        << ", dZ =  " << std::abs(tkRef->dz() - l1TkMuZ0) << '\n'
                        << "   L1TkMu: pT = " << l1TkMuPt << ", eta = " << l1TkMuEta << ", phi = " << l1TkMuPhi
                        << ", z0 = " << l1TkMuZ0 << '\n'
                        << "Track:  pT = " << tkRef->pt() << ", eta = " << tkRef->eta() << ", phi = " << tkRef->phi()
                        << ", z0 = " << tkRef->dz();
    }
  }  // End L1TkMuon Loop

  outputTracks->reserve(tracksToKeep.size());
  for (size_t idx : tracksToKeep) {
    reco::TrackRef tkRef(tkCollectionH, idx);
    outputTracks->push_back(*tkRef);
  }

  iEvent.put(std::move(outputTracks));
}

void MuonTracksSelectorFromL1TkMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("L1TkMuonInputCollection", edm::InputTag("l1tTkMuonsGmt"))->setComment("L1TkMuons to match");
  // L1Tk muon selection parameters
  desc.add<double>("L1TkMuMinPt", 0)->setComment("Minimum pT of the L1TkMuons to match");

  // Track selection parameters
  desc.add<edm::InputTag>("TrackInputCollection", edm::InputTag("generalTracks"))->setComment("Input tracks");
  desc.add<double>("trackMaxPtForMatch", 50.0)
      ->setComment("Maximum track pT for the curvature term. Above this pT, the quality uses only dR and dZ.");
  desc.add<double>("maxChi2", 9)
      ->setComment("Scale of the curvature chi2 between the track and the L1TkMuon in the quality");
  desc.add<double>("maxDr", 0.1)->setComment("Scale of dR between the track and the L1TkMuon in the quality");
  desc.add<double>("maxDz", 1.0)->setComment("Scale of dZ between the track and the L1TkMuon in the quality");
  desc.add<double>("maxCombinedQuality", 25.0)->setComment("Maximum combined quality of a match (lower is better)");
  desc.add<uint32_t>("nTracksToKeep", 1)
      ->setComment("Maximum number of tracks to keep for each L1TkMuon, from the quality ranking");
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonTracksSelectorFromL1TkMuon);
