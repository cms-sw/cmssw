/** \class HLTEgammaL1TMatchFilterRegional
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */
#include "HLTEgammaL1TMatchFilterRegional.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"

#define TWOPI 6.283185308
//
// constructors and destructor
//
HLTEgammaL1TMatchFilterRegional::HLTEgammaL1TMatchFilterRegional(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig) {
  candIsolatedTag_ = iConfig.getParameter<edm::InputTag>("candIsolatedTag");
  l1EGTag_ = iConfig.getParameter<edm::InputTag>(
      "l1IsolatedTag");  //will be renamed l1EGTag for step 2 of the new L1 migration
  candNonIsolatedTag_ = iConfig.getParameter<edm::InputTag>("candNonIsolatedTag");
  L1SeedFilterTag_ = iConfig.getParameter<edm::InputTag>("L1SeedFilterTag");
  l1JetsTag_ = iConfig.getParameter<edm::InputTag>(
      "l1CenJetsTag");  //will be renamed l1JetsTag for step 2 of the new L1 migration
  l1TausTag_ =
      iConfig.getParameter<edm::InputTag>("l1TausTag");  //will be renamed l1JetsTag for step 2 of the new L1 migration
  ncandcut_ = iConfig.getParameter<int>("ncandcut");
  doIsolated_ = iConfig.getParameter<bool>("doIsolated");
  matchToL1Jets_ = iConfig.getParameter<bool>("matchToL1Jets");
  matchToL1Taus_ = iConfig.getParameter<bool>("matchToL1Taus");
  region_eta_size_ = iConfig.getParameter<double>("region_eta_size");
  region_eta_size_ecap_ = iConfig.getParameter<double>("region_eta_size_ecap");
  region_phi_size_ = iConfig.getParameter<double>("region_phi_size");
  barrel_end_ = iConfig.getParameter<double>("barrel_end");
  endcap_end_ = iConfig.getParameter<double>("endcap_end");

  candIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candIsolatedTag_);
  if (!doIsolated_)
    candNonIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candNonIsolatedTag_);
  L1SeedFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(L1SeedFilterTag_);
}

HLTEgammaL1TMatchFilterRegional::~HLTEgammaL1TMatchFilterRegional() = default;

void HLTEgammaL1TMatchFilterRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candIsolatedTag", edm::InputTag("hltRecoIsolatedEcalCandidate"));
  desc.add<edm::InputTag>("l1IsolatedTag",
                          edm::InputTag("hltCaloStage2Digis"));  //rename for step 2 of the L1 migration
  desc.add<edm::InputTag>("candNonIsolatedTag", edm::InputTag("hltRecoNonIsolatedEcalCandidate"));
  desc.add<edm::InputTag>("l1NonIsolatedTag",
                          edm::InputTag("l1extraParticles", "NonIsolated"));  //drop for step 2 of the L1 migration
  desc.add<edm::InputTag>("L1SeedFilterTag", edm::InputTag("theL1SeedFilter"));
  desc.add<edm::InputTag>("l1CenJetsTag", edm::InputTag("hltCaloStage2Digis"));      //rename for step 2 of L1 migration
  desc.add<edm::InputTag>("l1TausTag", edm::InputTag("hltCaloStage2Digis", "Tau"));  //rename for step 2 of L1 migration
  desc.add<int>("ncandcut", 1);
  desc.add<bool>("doIsolated", true);
  desc.add<bool>("matchToL1Jets", true)->setComment("Also match to the L1T jets of the seed (Run-3 behaviour)");
  desc.add<bool>("matchToL1Taus", true)->setComment("Also match to the L1T taus of the seed (Run-3 behaviour)");
  // Matching regions: full widths of the eta/phi window around each L1T object. These values were tuned for
  // the Run-3 (Stage-2) L1T calo granularity. Phase-2 configurations currently reuse them; they need to be
  // updated for the Phase-2 L1T e/gamma position resolution (crystal-level barrel, HGCAL endcap).
  desc.add<double>("region_eta_size", 0.522)->setComment("Full eta width of the matching region in the barrel");
  desc.add<double>("region_eta_size_ecap", 1.0)->setComment("Full eta width of the matching region in the endcap");
  desc.add<double>("region_phi_size", 1.044)->setComment("Full phi width of the matching region");
  desc.add<double>("barrel_end", 1.4791)->setComment("|eta| of the HLT candidate separating barrel and endcap");
  desc.add<double>("endcap_end", 2.65)->setComment("Maximum |eta| of the HLT candidates");
  descriptions.add("HLTEgammaL1TMatchFilterRegional", desc);
}

// ------------ method called to produce the data  ------------
//configuration:
//doIsolated=true, only isolated superclusters are allowed to match isolated L1 seeds
//doIsolated=false, isolated superclusters are allowed to match either iso or non iso L1 seeds, non isolated superclusters are allowed only to match non-iso seeds. If no collection name is given for non-isolated superclusters, assumes the the isolated collection contains all (both iso + non iso) seeded superclusters.
bool HLTEgammaL1TMatchFilterRegional::hltFilter(edm::Event& iEvent,
                                                const edm::EventSetup& iSetup,
                                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // std::cout <<"runnr "<<iEvent.id().run()<<" event "<<iEvent.id().event()<<std::endl;
  using namespace trigger;
  //using namespace l1extra;

  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
    filterproduct.addCollectionTag(l1JetsTag_);
    filterproduct.addCollectionTag(l1TausTag_);
    filterproduct.addCollectionTag(candIsolatedTag_);
    if (!doIsolated_ && !candNonIsolatedTag_.label().empty()) {
      filterproduct.addCollectionTag(candNonIsolatedTag_);
    }
  }

  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByToken(candIsolatedToken_, recoIsolecalcands);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> L1SeedOutput;
  iEvent.getByToken(L1SeedFilterToken_, L1SeedOutput);

  // L1T objects of the seed: Run-3 (l1t::EGamma, l1t::Jet, l1t::Tau) or Phase-2 (l1t::P2GTCandidate)
  // refs, see hlt::getL1TCandidates
  auto const l1EGs = ::hlt::getL1TEGammas(*L1SeedOutput);
  auto const l1Jets = matchToL1Jets_ ? ::hlt::getL1TJets(*L1SeedOutput) : std::vector<reco::Candidate const*>();
  auto const l1Taus = matchToL1Taus_ ? ::hlt::getL1TTaus(*L1SeedOutput) : std::vector<reco::Candidate const*>();

  for (auto recoecalcand = recoIsolecalcands->begin(); recoecalcand != recoIsolecalcands->end(); recoecalcand++) {
    if (std::abs(recoecalcand->eta()) < endcap_end_) {
      //SC should be inside the ECAL fiducial volume

      //now EGamma is just one collection so automatically matches to Isolated and NonIsolated Seeds
      //it is assumed the HLTL1TSeed module fills it with the correct seeds
      bool matchedSCEG = matchedToL1Cand(l1EGs, recoecalcand->eta(), recoecalcand->phi());
      bool matchedSCJet = matchedToL1Cand(l1Jets, recoecalcand->eta(), recoecalcand->phi());
      bool matchedSCTau = matchedToL1Cand(l1Taus, recoecalcand->eta(), recoecalcand->phi());

      if (matchedSCEG || matchedSCJet || matchedSCTau) {
        n++;
        ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands,
                                                          distance(recoIsolecalcands->begin(), recoecalcand));
        filterproduct.addObject(TriggerCluster, ref);
      }  //end  matched check

    }  //end endcap fiduical check

  }  //end loop over all isolated RecoEcalCandidates

  //if doIsolated_ is false now run over the nonisolated superclusters and EG
  //however in the case we have a single collection of superclusters containing both iso L1 and non iso L1 seeded superclusters,
  //we do not have a non isolated collection of superclusters so we have to protect against that
  if (!doIsolated_ && !candNonIsolatedTag_.label().empty()) {
    edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
    iEvent.getByToken(candNonIsolatedToken_, recoNonIsolecalcands);

    for (auto recoecalcand = recoNonIsolecalcands->begin(); recoecalcand != recoNonIsolecalcands->end();
         recoecalcand++) {
      if (std::abs(recoecalcand->eta()) < endcap_end_) {
        bool matchedSCEG = matchedToL1Cand(l1EGs, recoecalcand->eta(), recoecalcand->phi());
        bool matchedSCJet = matchedToL1Cand(l1Jets, recoecalcand->eta(), recoecalcand->phi());
        bool matchedSCTau = matchedToL1Cand(l1Taus, recoecalcand->eta(), recoecalcand->phi());

        if (matchedSCEG || matchedSCJet || matchedSCTau) {
          n++;
          ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands,
                                                            distance(recoNonIsolecalcands->begin(), recoecalcand));
          filterproduct.addObject(TriggerCluster, ref);
        }  //end  matched check

      }  //end endcap fiduical check

    }  //end loop over all isolated RecoEcalCandidates
  }  //end doIsolatedCheck

  // filter decision
  bool accept(n >= ncandcut_);

  return accept;
}

bool HLTEgammaL1TMatchFilterRegional::matchedToL1Cand(const std::vector<reco::Candidate const*>& l1Cands,
                                                      const float scEta,
                                                      const float scPhi) const {
  for (auto const* l1Cand : l1Cands) {
    //ORCA matching method
    double etaBinLow = 0.;
    double etaBinHigh = 0.;
    if (std::abs(scEta) < barrel_end_) {
      etaBinLow = l1Cand->eta() - region_eta_size_ / 2.;
      etaBinHigh = etaBinLow + region_eta_size_;
    } else {
      etaBinLow = l1Cand->eta() - region_eta_size_ecap_ / 2.;
      etaBinHigh = etaBinLow + region_eta_size_ecap_;
    }

    float deltaphi = std::abs(scPhi - l1Cand->phi());
    if (deltaphi > TWOPI)
      deltaphi -= TWOPI;
    if (deltaphi > TWOPI / 2.)
      deltaphi = TWOPI - deltaphi;

    if (scEta < etaBinHigh && scEta > etaBinLow && deltaphi < region_phi_size_ / 2.) {
      return true;
    }
  }
  return false;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaL1TMatchFilterRegional);
