/** \class HLTMuonL3andL2PreFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing
 *  a simple filter to select L3 muons and L2 if they dont match with any L3
 * 
 *  Original author:  A. Soto <alejandro.soto.rodriguez@cern.ch>
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <vector>

class HLTMuonL3andL2PreFilter : public HLTFilter {
public:
  explicit HLTMuonL3andL2PreFilter(const edm::ParameterSet&);
  ~HLTMuonL3andL2PreFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs&) const override;

private:
  class MuonSelectionCuts {
  public:
    explicit MuonSelectionCuts(const edm::ParameterSet&);

    static void fillPSetDescription(edm::ParameterSetDescription&);

    const int min_N;                   // minimum number of muons to fire the trigger
    const double max_DXYBeamSpot;      // cutoff in dxy from the beamspot
    const double min_DXYBeamSpot;      // minimum cut on dxy from the beamspot
    const double max_Dz;               // dz cut
    const double min_DxySig;           // dxy significance cut
    const double nsigma_Pt;            // pt uncertainty margin (in number of sigmas)
    const double max_PtDifference;     // cutoff in maximum different between global track and tracker track
    const double min_TrackPt;          // cutoff in tracker track pt
    const double max_NormalizedChi2;   // cutoff in normalized chi2
    const double min_Pt;               // pt threshold in GeV
    const double max_Eta;              // Eta cut
    const int min_NmuonHits;           // cutoff in minumum number of chi2 hits
    const std::vector<int> min_Nhits;  // threshold on number of hits on muon
    const std::vector<double>
        absetaBins;  // |eta| bins for minNstations cut (#bins must match #minNstations cuts and #minNhits cuts)
    const std::vector<int> minNstations;  // minimum number of muon stations used
    const std::vector<int> minNchambers;  // minimum number of valid chambers
  };

  bool triggeredByPreviousLevel(const reco::RecoChargedCandidateRef&,
                                const std::vector<reco::RecoChargedCandidateRef>&) const;

  bool checkOverlap(const reco::RecoChargedCandidateRef&,
                    const std::vector<reco::RecoChargedCandidateRef>&,
                    const double) const;

  bool applySelection(const reco::RecoChargedCandidateRef&, const reco::BeamSpot&, const MuonSelectionCuts&) const;

  const edm::InputTag L3CandTag_;                                             // input tag identifying L3 muon container
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> L3CandToken_;  // token identifying L3 muon container

  const edm::InputTag L2CandTag_;                                             // input tag identifying L2 muon container
  const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> L2CandToken_;  // token identifying L2 muon container

  const edm::InputTag previousCandTag_;  // input tag identifying product contains muons passing the previous level
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>
      previousCandToken_;  // token identifying product contains muons passing the previous level

  const edm::InputTag beamspotTag_;                       // input tag identifying beamSpot container
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;  // token identifying beamSpot container

  const bool matchPreviousCand_;  // flag to match L2 or L3 candidates to the previous level
  const double maxDeltaR_L2L3_;   // maximum deltaR between L2 and L3 muon to consider them the same

  const MuonSelectionCuts cutParsL3_;  // container of all parameters to cut for L3
  const MuonSelectionCuts cutParsL2_;  // container of all parameters to cut for L2
};

HLTMuonL3andL2PreFilter::MuonSelectionCuts::MuonSelectionCuts(const edm::ParameterSet& iConfig)
    : min_N(iConfig.getParameter<int>("MinN")),
      max_DXYBeamSpot(iConfig.getParameter<double>("MaxDXYBeamSpot")),
      min_DXYBeamSpot(iConfig.getParameter<double>("MinDXYBeamSpot")),
      max_Dz(iConfig.getParameter<double>("MaxDz")),
      min_DxySig(iConfig.getParameter<double>("MinDxySig")),
      nsigma_Pt(iConfig.getParameter<double>("NSigmaPt")),
      max_PtDifference(iConfig.getParameter<double>("MaxPtDifference")),
      min_TrackPt(iConfig.getParameter<double>("MinTrackPt")),
      max_NormalizedChi2(iConfig.getParameter<double>("MaxNormalizedChi2")),
      min_Pt(iConfig.getParameter<double>("MinPt")),
      max_Eta(iConfig.getParameter<double>("MaxEta")),
      min_NmuonHits(iConfig.getParameter<int>("MinNmuonHits")),
      min_Nhits(iConfig.getParameter<std::vector<int>>("MinNhits")),
      absetaBins(iConfig.getParameter<std::vector<double>>("AbsEtaBins")),
      minNstations(iConfig.getParameter<std::vector<int>>("MinNstations")),
      minNchambers(iConfig.getParameter<std::vector<int>>("MinNchambers")) {}

HLTMuonL3andL2PreFilter::HLTMuonL3andL2PreFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      L3CandTag_(iConfig.getParameter<edm::InputTag>("L3CandTag")),
      L3CandToken_(consumes(L3CandTag_)),
      L2CandTag_(iConfig.getParameter<edm::InputTag>("L2CandTag")),
      L2CandToken_(consumes(L2CandTag_)),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      previousCandToken_(consumes(previousCandTag_)),
      beamspotTag_(iConfig.getParameter<edm::InputTag>("BeamSpotTag")),
      beamspotToken_(consumes(beamspotTag_)),
      matchPreviousCand_(iConfig.getParameter<bool>("MatchToPreviousCand")),
      maxDeltaR_L2L3_(iConfig.getParameter<double>("MaxDeltaRL2L3")),
      cutParsL3_(iConfig.getParameter<edm::ParameterSet>("L3CandSelection")),
      cutParsL2_(iConfig.getParameter<edm::ParameterSet>("L2CandSelection")) {
  // check consistency of parameters
  if (cutParsL2_.absetaBins.size() != cutParsL2_.minNstations.size() ||
      cutParsL2_.absetaBins.size() != cutParsL2_.minNchambers.size() ||
      cutParsL2_.absetaBins.size() != cutParsL2_.min_Nhits.size()) {
    throw cms::Exception("Configuration")
        << "error in ParameterSet \"L2CandSelection\": size of \"AbsEtaBins\" (" << cutParsL2_.absetaBins.size()
        << "), \"MinNstations\" (" << cutParsL2_.minNstations.size() << "), \"MinNchambers\" ("
        << cutParsL2_.minNchambers.size() << ") and \"MinNhits\" (" << cutParsL2_.min_Nhits.size() << ") differ";
  }
  if (cutParsL3_.absetaBins.size() != cutParsL3_.minNstations.size() ||
      cutParsL3_.absetaBins.size() != cutParsL3_.minNchambers.size() ||
      cutParsL3_.absetaBins.size() != cutParsL3_.min_Nhits.size()) {
    throw cms::Exception("Configuration")
        << "error in ParameterSet \"L3CandSelection\": size of \"AbsEtaBins\" (" << cutParsL3_.absetaBins.size()
        << "), \"MinNstations\" (" << cutParsL3_.minNstations.size() << "), \"MinNchambers\" ("
        << cutParsL3_.minNchambers.size() << ") and \"MinNhits\" (" << cutParsL3_.min_Nhits.size() << ") differ";
  }

  if (maxDeltaR_L2L3_ <= 0.) {
    throw cms::Exception("Configuration")
        << "invalid value for parameter \"MaxDeltaRL2L3\" (must be > 0): " << maxDeltaR_L2L3_;
  }
}

void HLTMuonL3andL2PreFilter::MuonSelectionCuts::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<int>("MinN", 1);
  desc.add<double>("MaxDXYBeamSpot", 9999.0);
  desc.add<double>("MinDXYBeamSpot", -1.0);
  desc.add<double>("MaxDz", 9999.0);
  desc.add<double>("MinDxySig", -1.0);
  desc.add<double>("NSigmaPt", 0.0);
  desc.add<double>("MaxPtDifference", 9999.0);
  desc.add<double>("MinTrackPt", 0.0);
  desc.add<double>("MaxNormalizedChi2", 9999.0);
  desc.add<double>("MinPt", 23.0);
  desc.add<double>("MaxEta", 2.5);
  desc.add<int>("MinNmuonHits", 0);
  desc.add<std::vector<int>>("MinNhits", {1, 0});
  desc.add<std::vector<double>>("AbsEtaBins", {1, 9999.});
  desc.add<std::vector<int>>("MinNstations", {1, 1});
  desc.add<std::vector<int>>("MinNchambers", {1, 0});
};

void HLTMuonL3andL2PreFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("L3CandTag", edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("L2CandTag", edm::InputTag("hltL2MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag(""));
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<bool>("MatchToPreviousCand", true);
  desc.add<double>("MaxDeltaRL2L3", 0.05);

  edm::ParameterSetDescription descMuonSelL2;
  MuonSelectionCuts::fillPSetDescription(descMuonSelL2);
  desc.add<edm::ParameterSetDescription>("L2CandSelection", descMuonSelL2);

  edm::ParameterSetDescription descMuonSelL3;
  MuonSelectionCuts::fillPSetDescription(descMuonSelL3);
  desc.add<edm::ParameterSetDescription>("L3CandSelection", descMuonSelL3);

  descriptions.addWithDefaultLabel(desc);
}

bool HLTMuonL3andL2PreFilter::hltFilter(edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.
  if (saveTags()) {
    filterproduct.addCollectionTag(L3CandTag_);
    filterproduct.addCollectionTag(L2CandTag_);
  }
  // get hold of trks
  auto const L3mucands = iEvent.getHandle(L3CandToken_);
  auto const L2mucands = iEvent.getHandle(L2CandToken_);
  std::vector<reco::RecoChargedCandidateRef> vcands;
  if (matchPreviousCand_) {
    auto const& previousLevelCands = iEvent.get(previousCandToken_);
    previousLevelCands.getObjects(trigger::TriggerMuon, vcands);
  }

  auto const& beamSpot = iEvent.get(beamspotToken_);

  // Number of objects passing the L3 Trigger:
  int nL3 = 0;
  for (unsigned int iMuL3 = 0; iMuL3 < L3mucands->size(); iMuL3++) {
    reco::RecoChargedCandidateRef L3cand(L3mucands, iMuL3);
    LogDebug("HLTMuonL3andL2PreFilter") << "L3cand isNonnull " << L3cand.isNonnull();
    // did this candidate triggered at previous stage.
    if (matchPreviousCand_ && !triggeredByPreviousLevel(L3cand, vcands))
      continue;
    // apply all the cuts
    if (!applySelection(L3cand, beamSpot, cutParsL3_))
      continue;
    nL3++;
    filterproduct.addObject(trigger::TriggerMuon, L3cand);
  }
  // get the L3 muons that passed the selection
  std::vector<reco::RecoChargedCandidateRef> passingL3mucands;
  filterproduct.getObjects(trigger::TriggerMuon, passingL3mucands);

  // Number of objects passing the L2 Trigger:
  int nL2 = 0;
  auto const maxDeltaR2_L2L3 = maxDeltaR_L2L3_ * maxDeltaR_L2L3_;
  for (unsigned int iMuL2 = 0; iMuL2 < L2mucands->size(); iMuL2++) {
    reco::RecoChargedCandidateRef L2cand(L2mucands, iMuL2);
    LogDebug("HLTMuonL3andL2PreFilter") << "L2cand isNonnull " << L2cand.isNonnull();
    // did this candidate triggered at previous stage.
    if (matchPreviousCand_ && !triggeredByPreviousLevel(L2cand, vcands))
      continue;
    // check if L2 and L3 candidates are the same
    if (checkOverlap(L2cand, passingL3mucands, maxDeltaR2_L2L3)) {
      continue;
    }
    // apply all the cuts
    if (!applySelection(L2cand, beamSpot, cutParsL2_))
      continue;
    nL2++;
    filterproduct.addObject(trigger::TriggerMuon, L2cand);
  }

  // filter decision
  const bool acceptL3(nL3 >= cutParsL3_.min_N);
  const bool acceptL2(nL2 >= cutParsL2_.min_N);

  LogDebug("HLTMuonL3andL2PreFilter") << " >>>>> Result of HLTMuonL3andL2PreFilter is " << (acceptL3 && acceptL2)
                                      << ", number of muons passing thresholds= " << nL2 + nL3;

  return acceptL2 && acceptL3;
}

bool HLTMuonL3andL2PreFilter::triggeredByPreviousLevel(const reco::RecoChargedCandidateRef& candref,
                                                       const std::vector<reco::RecoChargedCandidateRef>& vcands) const {
  return (std::find(vcands.begin(), vcands.end(), candref) != vcands.end());
}

bool HLTMuonL3andL2PreFilter::checkOverlap(
    const reco::RecoChargedCandidateRef& r1,
    const std::vector<reco::RecoChargedCandidateRef>& R2,  // vector with the recoCharged candidates from L3
    const double maxDeltaR2) const {
  for (auto const& r2 : R2) {
    if (reco::deltaR2(*r1, *r2) < maxDeltaR2)
      return true;
  }
  return false;
}

bool HLTMuonL3andL2PreFilter::applySelection(const reco::RecoChargedCandidateRef& candidate,
                                             const reco::BeamSpot& beamSpot,
                                             const MuonSelectionCuts& cutPars) const {
  // eta cut
  auto const candAbsEta = std::abs(candidate->eta());
  if (candAbsEta > cutPars.max_Eta)
    return false;

  reco::TrackRef const& tk = candidate->track();
  LogDebug("HLTMuonL3andL2PreFilter") << " Muon in loop, q*pt= " << tk->charge() * tk->pt() << " ("
                                      << candidate->charge() * candidate->pt() << ") "
                                      << ", eta= " << tk->eta() << " (" << candidate->eta() << ") "
                                      << ", hits= " << tk->numberOfValidHits() << ", d0= " << tk->d0()
                                      << ", dz= " << tk->dz();

  // normalizedChi2 cut
  if (tk->normalizedChi2() > cutPars.max_NormalizedChi2)
    return false;

  // number of eta bins for cut on number of stations
  const auto nAbsetaBins = cutPars.absetaBins.size();

  // cut on number of stations
  bool failNstations(false), failNhits(false), failNchambers(false);
  for (unsigned int i = 0; i < nAbsetaBins; ++i) {
    if (candAbsEta < cutPars.absetaBins[i]) {
      if (candidate->track()->hitPattern().muonStationsWithAnyHits() < cutPars.minNstations[i]) {
        failNstations = true;
      }
      if (candidate->track()->numberOfValidHits() < cutPars.min_Nhits[i]) {
        failNhits = true;
      }
      if ((candidate->track()->hitPattern().dtStationsWithAnyHits() +
           candidate->track()->hitPattern().cscStationsWithAnyHits()) < cutPars.minNchambers[i]) {
        failNchambers = true;
      }
      break;
    }
  }
  if (failNstations || failNhits || failNchambers)
    return false;

  // dz cut (impact parameter of the muon in the z direction computed with respect to the beamspot position)
  auto const vz = candidate->vz() - beamSpot.z0();
  auto const vx = candidate->vx() - beamSpot.x0();
  auto const vy = candidate->vy() - beamSpot.y0();
  if (candidate->pt() <= 0 or
      (std::abs(vz - (vx * candidate->px() + vy * candidate->py()) / (candidate->pt() * candidate->pt()) *
                         candidate->pz()) > cutPars.max_Dz))
    return false;

  // dxy significance cut (safeguard against bizarre values)
  auto const absDxy = std::abs(tk->dxy(beamSpot.position()));
  if (cutPars.min_DxySig > 0 && (tk->dxyError() <= 0 || (absDxy < cutPars.min_DxySig * tk->dxyError())))
    return false;

  // dxy beamspot cut
  if (absDxy > cutPars.max_DXYBeamSpot || absDxy < cutPars.min_DXYBeamSpot)
    return false;

  // min muon hits cut
  const reco::HitPattern& trackHits = tk->hitPattern();
  if (trackHits.numberOfValidMuonHits() < cutPars.min_NmuonHits)
    return false;

  // pt difference cut
  auto const candPt = candidate->pt();
  auto const trackPt = tk->pt();

  if (std::abs(candPt - trackPt) > cutPars.max_PtDifference)
    return false;

  // track pt cut
  if (trackPt < cutPars.min_TrackPt)
    return false;

  // pt threshold cut
  auto const err0 = tk->error(0);
  auto const abspar0 = std::abs(tk->parameter(0));
  // if abspar0 > 0, convert 50% efficiency threshold to 90% efficiency threshold
  auto const ptLx = abspar0 > 0 ? candPt + cutPars.nsigma_Pt * err0 / abspar0 * candPt : candPt;
  if (ptLx < cutPars.min_Pt)
    return false;

  return true;
}

DEFINE_FWK_MODULE(HLTMuonL3andL2PreFilter);
