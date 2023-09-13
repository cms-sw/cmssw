/*
 * \class TauTagFilter
 *
 * Filter tau candidates based on tagger scores.
 *
 * \author Konstantin Androsov, EPFL and ETHZ
 */

#include "DataFormats/BTauReco/interface/JetTag.h"
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

  explicit TauTagFilter(const edm::ParameterSet& cfg)
      : HLTFilter(cfg),
        nExpected_(cfg.getParameter<int>("nExpected")),
        tausSrc_(cfg.getParameter<edm::InputTag>("taus")),
        tausToken_(consumes<TauCollection>(tausSrc_)),
        tauTagsToken_(consumes<TauTagCollection>(cfg.getParameter<edm::InputTag>("tauTags"))),
        tauPtCorrToken_(mayConsume<TauTagCollection>(cfg.getParameter<edm::InputTag>("tauPtCorr"))),
        selector_(cfg.getParameter<std::string>("selection")),
        minPt_(cfg.getParameter<double>("minPt")),
        maxEta_(cfg.getParameter<double>("maxEta")),
        usePtCorr_(cfg.getParameter<bool>("usePtCorr")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<int>("nExpected", 2)->setComment("number of expected taus per event");
    desc.add<edm::InputTag>("taus", edm::InputTag(""))->setComment("input collection of taus");
    desc.add<edm::InputTag>("tauTags", edm::InputTag(""))->setComment("input collection of tau tagger scores");
    desc.add<edm::InputTag>("tauPtCorr", edm::InputTag(""))->setComment("input collection of multiplicative tau pt corrections");
    desc.add<std::string>("selector", "0")->setComment("selection formula");
    desc.add<double>("minPt", 20)->setComment("minimal tau pt");
    desc.add<double>("maxEta", 2.5)->setComment("maximal tau abs(eta)");
    desc.add<bool>("usePtCorr", false)->setComment("use multiplicative tau pt corrections");
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

    const auto& tauTags = event.get(tauTagsToken_);
    const TauTagCollection* tauPtCorr = nullptr;
    if (usePtCorr_)
      tauPtCorr = &event.get(tauPtCorrToken_);

    if (taus.size() != tauTags.size())
      throw cms::Exception("Inconsistent Data", "PNetTauTagFilter::hltFilter") << "taus.size() != tauTags.size()";
    if (usePtCorr_ && taus.size() != tauPtCorr->size())
      throw cms::Exception("Inconsistent Data", "PNetTauTagFilter::hltFilter") << "taus.size() != tauPtCorr.size()";

    for (size_t tau_idx = 0; tau_idx < taus.size(); ++tau_idx) {
      const auto& tau = taus[tau_idx];
      double pt = tau.pt();
      if(usePtCorr_)
        pt *= (*tauPtCorr)[tau_idx].second;
      const double eta = std::abs(tau.eta());

      if (pt > minPt_ && eta < maxEta_) {
        const double tag = tauTags[tau_idx].second;
        const double tag_thr = selector_(tau);
        if (tag > tag_thr) {
          filterproduct.addObject(nTauPassed, TauRef(tausHandle, tau_idx));
          nTauPassed++;
        }
      }
    }

    return nTauPassed >= nExpected_;
  }

private:
  const int nExpected_;
  const edm::InputTag tausSrc_;
  const edm::EDGetTokenT<TauCollection> tausToken_;
  const edm::EDGetTokenT<TauTagCollection> tauTagsToken_, tauPtCorrToken_;
  const Selector selector_;
  const double minPt_, maxEta_;
  const bool usePtCorr_;
};

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauTagFilter);
