/**  \class HLTMuonL2SelectorForL3IO
 * 
 *   L2 muon selector for L3 IO:
 *   finds L2 muons not previous converted into (good) L3 muons
 *
 *   \author  Benjamin Radburn-Smith, Santiago Folgueras - Purdue University
 */

#include "RecoMuon/L3TrackFinder/interface/HLTMuonL2SelectorForL3IO.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

/// constructor with config
HLTMuonL2SelectorForL3IO::HLTMuonL2SelectorForL3IO(const edm::ParameterSet& iConfig)
    : l2Src_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("l2Src"))),
      l3OISrc_(consumes<reco::RecoChargedCandidateCollection>(iConfig.getParameter<edm::InputTag>("l3OISrc"))),
      l3linkToken_(consumes<reco::MuonTrackLinksCollection>(iConfig.getParameter<edm::InputTag>("InputLinks"))),
      applyL3Filters_(iConfig.getParameter<bool>("applyL3Filters")),
      max_NormalizedChi2_(iConfig.getParameter<double>("MaxNormalizedChi2")),
      max_PtDifference_(iConfig.getParameter<double>("MaxPtDifference")),
      min_Nhits_(iConfig.getParameter<int>("MinNhits")),
      min_NmuonHits_(iConfig.getParameter<int>("MinNmuonHits")) {
  LogTrace("Muon|RecoMuon|HLTMuonL2SelectorForL3IO") << "constructor called";
  produces<reco::TrackCollection>();
}

/// destructor
HLTMuonL2SelectorForL3IO::~HLTMuonL2SelectorForL3IO() {}

/// create collection of L2 muons not already reconstructed as L3 muons
void HLTMuonL2SelectorForL3IO::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::string metname = "Muon|RecoMuon|HLTMuonL2SelectorForL3IO";

  //  IN:
  edm::Handle<reco::TrackCollection> l2muonH;
  iEvent.getByToken(l2Src_, l2muonH);

  edm::Handle<reco::RecoChargedCandidateCollection> l3muonH;
  iEvent.getByToken(l3OISrc_, l3muonH);

  // Read Links collection:
  edm::Handle<reco::MuonTrackLinksCollection> links;
  iEvent.getByToken(l3linkToken_, links);

  //	OUT:
  std::unique_ptr<reco::TrackCollection> result(new reco::TrackCollection());

  for (unsigned int il2 = 0; il2 != l2muonH->size(); ++il2) {
    reco::TrackRef l2muRef(l2muonH, il2);
    bool re_do_this_L2 = true;

    for (unsigned int il3 = 0; il3 != l3muonH->size(); ++il3) {
      reco::RecoChargedCandidateRef cand(l3muonH, il3);
      reco::TrackRef tk = cand->track();

      bool useThisLink = false;
      for (unsigned int l(0); l < links->size() && !useThisLink; ++l) {
        const reco::MuonTrackLinks* link = &links->at(l);

        // Check if the L3 link matches the L3 candidate
        const reco::Track& globalTrack = *link->globalTrack();
        float dR2 = deltaR2(tk->eta(), tk->phi(), globalTrack.eta(), globalTrack.phi());
        if (dR2 < 0.02 * 0.02 and std::abs(tk->pt() - globalTrack.pt()) < 0.001 * tk->pt()) {
          useThisLink = true;
        }

        if (!useThisLink)
          continue;

        // Check whether the stand-alone track matches a L2, if not, we will re-use this L2
        const reco::TrackRef staTrack = link->standAloneTrack();
        if (l2muRef == staTrack)
          re_do_this_L2 = false;

        // Check the quality of the reconstructed L3, if poor quality, we will re-use this L2
        if (staTrack == l2muRef && applyL3Filters_) {
          re_do_this_L2 = true;
          const reco::Track& globalTrack = *link->globalTrack();
          if (globalTrack.numberOfValidHits() < min_Nhits_)
            continue;  // cut on number of hits
          if (globalTrack.normalizedChi2() > max_NormalizedChi2_)
            continue;  //normalizedChi2 cut
          if (globalTrack.hitPattern().numberOfValidMuonHits() < min_NmuonHits_)
            continue;  //min muon hits cut
          if (std::abs(globalTrack.pt() - l2muRef->pt()) > max_PtDifference_ * globalTrack.pt())
            continue;  // pt difference
          re_do_this_L2 = false;
        }
      }
    }
    if (re_do_this_L2)
      result->push_back(*l2muRef);  // used the L2 if no L3 if matched or if the matched L3 has poor quality cuts.
  }
  iEvent.put(std::move(result));
}

void HLTMuonL2SelectorForL3IO::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l2Src", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
  desc.add<edm::InputTag>("l3OISrc", edm::InputTag("hltNewOIL3MuonCandidates"));
  desc.add<edm::InputTag>("InputLinks", edm::InputTag("hltNewOIL3MuonsLinksCombination"));
  desc.add<bool>("applyL3Filters", true);
  desc.add<int>("MinNhits", 1);
  desc.add<double>("MaxNormalizedChi2", 20.0);
  desc.add<int>("MinNmuonHits", 0);
  desc.add<double>("MaxPtDifference", 999.0);  //relative difference
  descriptions.add("HLTMuonL2SelectorForL3IO", desc);
}
