/*
 * \class TauTagFilter
 *
 * Filter tau candidates based on tagger scores.
 *
 * \author Konstantin Androsov, EPFL and ETHZ
 */

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "RecoTauTag/RecoTau/interface/TauWPThreshold.h"

class TauTagFilter : public HLTFilter {
public:
  using TauCollection = reco::PFJetCollection;
  using TauTagCollection = reco::JetTagCollection;
  using TauRef = reco::PFJetRef;
  using Selector = tau::TauWPThreshold;
  using LorentzVectorM = math::PtEtaPhiMLorentzVector;

  explicit TauTagFilter(const edm::ParameterSet& cfg)
      : HLTFilter(cfg),
        nExpected_(cfg.getParameter<int>("nExpected")),
        tausSrc_(cfg.getParameter<edm::InputTag>("taus")),
        tausToken_(consumes<TauCollection>(tausSrc_)),
        tauTagsToken_(consumes<TauTagCollection>(cfg.getParameter<edm::InputTag>("tauTags"))),
        tauPtCorrToken_(mayConsume<TauTagCollection>(cfg.getParameter<edm::InputTag>("tauPtCorr"))),
        seedsSrc_(mayConsume<trigger::TriggerFilterObjectWithRefs>(cfg.getParameter<edm::InputTag>("seeds"))),
        seedTypes_(cfg.getParameter<std::vector<int>>("seedTypes")),
        selector_(cfg.getParameter<std::string>("selection")),
        minPt_(cfg.getParameter<double>("minPt")),
        maxEta_(cfg.getParameter<double>("maxEta")),
        usePtCorr_(cfg.getParameter<bool>("usePtCorr")),
        matchWithSeeds_(cfg.getParameter<bool>("matchWithSeeds") && cfg.getParameter<double>("matchingdR") >= 0),
        matchingdR2_(std::pow(cfg.getParameter<double>("matchingdR"), 2)) {
    if (cfg.getParameter<bool>("matchWithSeeds") && cfg.getParameter<double>("matchingdR") < 0)
      edm::LogWarning("TauTagFilter") << "Matching with seeds is disabled because matchingdR < 0";

    extractMomenta();  // checking that all seed types are supported
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<int>("nExpected", 2)->setComment("number of expected taus per event");
    desc.add<edm::InputTag>("taus", edm::InputTag(""))->setComment("input collection of taus");
    desc.add<edm::InputTag>("tauTags", edm::InputTag(""))->setComment("input collection of tau tagger scores");
    desc.add<edm::InputTag>("tauPtCorr", edm::InputTag(""))
        ->setComment("input collection of multiplicative tau pt corrections");
    desc.add<edm::InputTag>("seeds", edm::InputTag(""))->setComment("input collection of seeds");
    desc.add<std::vector<int>>("seedTypes",
                               {trigger::TriggerL1Tau, trigger::TriggerL1Jet, trigger::TriggerTau, trigger::TriggerJet})
        ->setComment("list of seed object types");
    desc.add<std::string>("selection", "0")->setComment("selection formula");
    desc.add<double>("minPt", 20)->setComment("minimal tau pt");
    desc.add<double>("maxEta", 2.5)->setComment("maximal tau abs(eta)");
    desc.add<bool>("usePtCorr", false)->setComment("use multiplicative tau pt corrections");
    desc.add<bool>("matchWithSeeds", false)->setComment("apply match with seeds");
    desc.add<double>("matchingdR", 0.5)->setComment("deltaR for matching with seeds");
    descriptions.addWithDefaultLabel(desc);
  }

  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& eventsetup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override {
    if (saveTags())
      filterproduct.addCollectionTag(tausSrc_);

    int nTauPassed = 0;

    const auto tausHandle = event.getHandle(tausToken_);
    const auto& taus = *tausHandle;

    const std::vector<LorentzVectorM> seed_p4s = extractMomenta(&event);
    auto hasMatch = [&](const LorentzVectorM& p4) {
      for (const auto& seed_p4 : seed_p4s) {
        if (reco::deltaR2(p4, seed_p4) < matchingdR2_)
          return true;
      }
      return false;
    };

    const auto& tauTags = event.get(tauTagsToken_);
    const TauTagCollection* tauPtCorr = nullptr;
    if (usePtCorr_)
      tauPtCorr = &event.get(tauPtCorrToken_);

    if (taus.size() != tauTags.size())
      throw cms::Exception("Inconsistent Data", "TauTagFilter::hltFilter") << "taus.size() != tauTags.size()";
    if (usePtCorr_ && taus.size() != tauPtCorr->size())
      throw cms::Exception("Inconsistent Data", "TauTagFilter::hltFilter") << "taus.size() != tauPtCorr.size()";

    for (size_t tau_idx = 0; tau_idx < taus.size(); ++tau_idx) {
      const auto& tau = taus[tau_idx];
      double pt = tau.pt();
      if (usePtCorr_)
        pt *= (*tauPtCorr)[tau_idx].second;
      const double eta = std::abs(tau.eta());
      if (pt > minPt_ && eta < maxEta_ && (!matchWithSeeds_ || hasMatch(tau.polarP4()))) {
        const double tag = tauTags[tau_idx].second;
        const double tag_thr = selector_(tau);
        if (tag > tag_thr) {
          filterproduct.addObject(trigger::TriggerTau, TauRef(tausHandle, tau_idx));
          nTauPassed++;
        }
      }
    }

    return nTauPassed >= nExpected_;
  }

private:
  std::vector<LorentzVectorM> extractMomenta(const edm::Event* event = nullptr) const {
    std::vector<LorentzVectorM> seed_p4s;
    if (matchWithSeeds_) {
      const trigger::TriggerFilterObjectWithRefs* seeds = nullptr;
      if (event)
        seeds = &event->get(seedsSrc_);
      for (const int seedType : seedTypes_) {
        if (seedType == trigger::TriggerL1Tau) {
          extractMomenta<l1t::TauVectorRef>(seeds, seedType, seed_p4s);
        } else if (seedType == trigger::TriggerL1Jet) {
          extractMomenta<l1t::JetVectorRef>(seeds, seedType, seed_p4s);
        } else if (seedType == trigger::TriggerTau) {
          extractMomenta<std::vector<reco::PFTauRef>>(seeds, seedType, seed_p4s);
        } else if (seedType == trigger::TriggerJet) {
          extractMomenta<std::vector<reco::PFJetRef>>(seeds, seedType, seed_p4s);
        } else
          throw cms::Exception("Invalid seed type", "TauTagFilter::extractMomenta")
              << "Unsupported seed type: " << seedType;
      }
    }
    return seed_p4s;
  }

  template <typename Collection>
  static void extractMomenta(const trigger::TriggerRefsCollections* triggerObjects,
                             int objType,
                             std::vector<LorentzVectorM>& p4s) {
    if (triggerObjects) {
      Collection objects;
      triggerObjects->getObjects(objType, objects);
      for (const auto& obj : objects)
        p4s.push_back(obj->polarP4());
    }
  }

private:
  const int nExpected_;
  const edm::InputTag tausSrc_;
  const edm::EDGetTokenT<TauCollection> tausToken_;
  const edm::EDGetTokenT<TauTagCollection> tauTagsToken_, tauPtCorrToken_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> seedsSrc_;
  const std::vector<int> seedTypes_;
  const Selector selector_;
  const double minPt_, maxEta_;
  const bool usePtCorr_;
  const bool matchWithSeeds_;
  const double matchingdR2_;
};

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauTagFilter);
