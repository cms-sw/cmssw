/** \class HLTMuonL3SimplePreFilter
 *
 * See header file for documentation
 *
 *
 */

#include "HLTMuonL3SimplePreFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <iostream>

//
// constructors and destructor
//
using namespace std;
using namespace edm;
using namespace trigger;
using namespace reco;

HLTMuonL3SimplePreFilter::HLTMuonL3SimplePreFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      candTag_(iConfig.getParameter<edm::InputTag>("CandTag")),
      previousCandTag_(iConfig.getParameter<edm::InputTag>("PreviousCandTag")),
      beamspotTag_(iConfig.getParameter<edm::InputTag>("BeamSpotTag")),
      min_N_(iConfig.getParameter<int>("MinN")),
      max_Eta_(iConfig.getParameter<double>("MaxEta")),
      min_Nhits_(iConfig.getParameter<int>("MinNhits")),
      max_Dz_(iConfig.getParameter<double>("MaxDz")),
      min_DxySig_(iConfig.getParameter<double>("MinDxySig")),
      min_Pt_(iConfig.getParameter<double>("MinPt")),
      nsigma_Pt_(iConfig.getParameter<double>("NSigmaPt")),
      max_NormalizedChi2_(iConfig.getParameter<double>("MaxNormalizedChi2")),
      max_DXYBeamSpot_(iConfig.getParameter<double>("MaxDXYBeamSpot")),
      min_DXYBeamSpot_(iConfig.getParameter<double>("MinDXYBeamSpot")),
      min_NmuonHits_(iConfig.getParameter<int>("MinNmuonHits")),
      max_PtDifference_(iConfig.getParameter<double>("MaxPtDifference")),
      min_TrackPt_(iConfig.getParameter<double>("MinTrackPt")),
      matchPreviousCand_(iConfig.getParameter<bool>("MatchToPreviousCand")) {
  candToken_ = consumes<reco::RecoChargedCandidateCollection>(candTag_);
  previousCandToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_);
  beamspotToken_ = consumes<reco::BeamSpot>(beamspotTag_);
}

HLTMuonL3SimplePreFilter::~HLTMuonL3SimplePreFilter() = default;

//
// member functions
//
void HLTMuonL3SimplePreFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag", edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag(""));
  desc.add<int>("MinN", 1);
  desc.add<double>("MaxEta", 2.5);
  desc.add<int>("MinNhits", 0);
  desc.add<double>("MaxDz", 9999.0);
  desc.add<double>("MinDxySig", -1.0);
  desc.add<double>("MinPt", 3.0);
  desc.add<double>("NSigmaPt", 0.0);
  desc.add<double>("MaxNormalizedChi2", 9999.0);
  desc.add<double>("MaxDXYBeamSpot", 9999.0);
  desc.add<double>("MinDXYBeamSpot", -1.0);
  desc.add<int>("MinNmuonHits", 0);
  desc.add<double>("MaxPtDifference", 9999.0);
  desc.add<double>("MinTrackPt", 0.0);
  desc.add<bool>("MatchToPreviousCand", true);
  descriptions.add("hltMuonL3SimplePreFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTMuonL3SimplePreFilter::hltFilter(edm::Event& iEvent,
                                         const edm::EventSetup& iSetup,
                                         trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  if (saveTags())
    filterproduct.addCollectionTag(candTag_);

  // get hold of trks
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByToken(candToken_, mucands);
  Handle<TriggerFilterObjectWithRefs> previousLevelCands;
  iEvent.getByToken(previousCandToken_, previousLevelCands);
  vector<RecoChargedCandidateRef> vcands;
  if (previousLevelCands.isValid()) {
    previousLevelCands->getObjects(TriggerMuon, vcands);
  }

  Handle<BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamspotToken_, recoBeamSpotHandle);

  // Number of objects passing the L3 Trigger:
  int n = 0;
  for (unsigned int iMu = 0; iMu < mucands->size(); iMu++) {
    RecoChargedCandidateRef cand(mucands, iMu);
    LogDebug("HLTMuonL3SimplePreFilter") << "cand isNonnull " << cand.isNonnull();

    //did this candidate triggered at previous stage.
    if (matchPreviousCand_ && !triggerdByPreviousLevel(cand, vcands))
      continue;

    if (std::abs(cand->eta()) > max_Eta_)
      continue;

    TrackRef tk = cand->track();
    LogDebug("HLTMuonL3SimplePreFilter") << " Muon in loop, q*pt= " << tk->charge() * tk->pt() << " ("
                                         << cand->charge() * cand->pt() << ") "
                                         << ", eta= " << tk->eta() << " (" << cand->eta() << ") "
                                         << ", hits= " << tk->numberOfValidHits() << ", d0= " << tk->d0()
                                         << ", dz= " << tk->dz();

    // cut on number of hits
    if (tk->numberOfValidHits() < min_Nhits_)
      continue;

    //normalizedChi2 cut
    if (tk->normalizedChi2() > max_NormalizedChi2_)
      continue;

    if (recoBeamSpotHandle.isValid()) {
      const BeamSpot& beamSpot = *recoBeamSpotHandle;

      //dz cut
      if (std::abs((cand->vz() - beamSpot.z0()) -
                   ((cand->vx() - beamSpot.x0()) * cand->px() + (cand->vy() - beamSpot.y0()) * cand->py()) /
                       cand->pt() * cand->pz() / cand->pt()) > max_Dz_)
        continue;

      // dxy significance cut (safeguard against bizarre values)
      if (min_DxySig_ > 0 &&
          (tk->dxyError() <= 0 || std::abs(tk->dxy(beamSpot.position()) / tk->dxyError()) < min_DxySig_))
        continue;

      //dxy beamspot cut
      float absDxy = std::abs(tk->dxy(beamSpot.position()));
      if (absDxy > max_DXYBeamSpot_ || absDxy < min_DXYBeamSpot_)
        continue;
    }

    //min muon hits cut
    const reco::HitPattern& trackHits = tk->hitPattern();
    if (trackHits.numberOfValidMuonHits() < min_NmuonHits_)
      continue;

    //pt difference cut
    double candPt = cand->pt();
    double trackPt = tk->pt();

    if (std::abs(candPt - trackPt) > max_PtDifference_)
      continue;

    //track pt cut
    if (trackPt < min_TrackPt_)
      continue;

    // Pt threshold cut
    double pt = cand->pt();
    double err0 = tk->error(0);
    double abspar0 = std::abs(tk->parameter(0));
    double ptLx = pt;
    // convert 50% efficiency threshold to 90% efficiency threshold
    if (abspar0 > 0)
      ptLx += nsigma_Pt_ * err0 / abspar0 * pt;
    LogTrace("HLTMuonL3SimplePreFilter") << " ...Muon in loop, trackkRef pt= " << tk->pt() << ", ptLx= " << ptLx
                                         << " cand pT " << cand->pt();
    if (ptLx < min_Pt_)
      continue;

    n++;
    filterproduct.addObject(TriggerMuon, cand);
  }  //for iMu

  // filter decision
  const bool accept(n >= min_N_);

  LogDebug("HLTMuonL3SimplePreFilter") << " >>>>> Result of HLTMuonL3PreFilter is " << accept
                                       << ", number of muons passing thresholds= " << n;

  return accept;
}

bool HLTMuonL3SimplePreFilter::triggerdByPreviousLevel(const reco::RecoChargedCandidateRef& candref,
                                                       const std::vector<reco::RecoChargedCandidateRef>& vcands) {
  unsigned int i = 0;
  unsigned int i_max = vcands.size();
  for (; i != i_max; ++i) {
    if (candref == vcands[i])
      return true;
  }

  return false;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMuonL3SimplePreFilter);
