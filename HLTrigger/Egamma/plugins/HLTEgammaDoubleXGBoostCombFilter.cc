#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include <vector>
#include <memory>

namespace edm {
  class ConfigurationDescriptions;
}

class HLTEgammaDoubleXGBoostCombFilter : public HLTFilter {
public:
  explicit HLTEgammaDoubleXGBoostCombFilter(edm::ParameterSet const&);
  ~HLTEgammaDoubleXGBoostCombFilter() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& setup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  const double highMassCut_;
  const std::vector<double> leadCutHighMass1_;
  const std::vector<double> subCutHighMass1_;
  const std::vector<double> leadCutHighMass2_;
  const std::vector<double> subCutHighMass2_;
  const std::vector<double> leadCutHighMass3_;
  const std::vector<double> subCutHighMass3_;

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candToken_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> mvaToken_;
};

HLTEgammaDoubleXGBoostCombFilter::HLTEgammaDoubleXGBoostCombFilter(edm::ParameterSet const& config)
    : HLTFilter(config),
      highMassCut_(config.getParameter<double>("highMassCut")),
      leadCutHighMass1_(config.getParameter<std::vector<double>>("leadCutHighMass1")),
      subCutHighMass1_(config.getParameter<std::vector<double>>("subCutHighMass1")),
      leadCutHighMass2_(config.getParameter<std::vector<double>>("leadCutHighMass2")),
      subCutHighMass2_(config.getParameter<std::vector<double>>("subCutHighMass2")),
      leadCutHighMass3_(config.getParameter<std::vector<double>>("leadCutHighMass3")),
      subCutHighMass3_(config.getParameter<std::vector<double>>("subCutHighMass3")),
      candToken_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("candTag"))),
      mvaToken_(consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("mvaPhotonTag"))) {}

void HLTEgammaDoubleXGBoostCombFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);

  desc.add<double>("highMassCut", 90.0);
  desc.add<std::vector<double>>("leadCutHighMass1", {0.92, 0.95});
  desc.add<std::vector<double>>("subCutHighMass1", {0.02, 0.04});
  desc.add<std::vector<double>>("leadCutHighMass2", {0.85, 0.85});
  desc.add<std::vector<double>>("subCutHighMass2", {0.04, 0.08});
  desc.add<std::vector<double>>("leadCutHighMass3", {0.30, 0.50});
  desc.add<std::vector<double>>("subCutHighMass3", {0.14, 0.20});

  desc.add<edm::InputTag>("candTag", edm::InputTag("hltEgammaCandidatesUnseeded"));
  desc.add<edm::InputTag>("mvaPhotonTag", edm::InputTag("PhotonXGBoostProducer"));

  descriptions.addWithDefaultLabel(desc);
}

bool HLTEgammaDoubleXGBoostCombFilter::hltFilter(edm::Event& event,
                                                 const edm::EventSetup& setup,
                                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  const auto& recCollection = event.getHandle(candToken_);
  const auto& mvaMap = event.getHandle(mvaToken_);

  // Lambda to evaluate pair cuts
  auto passesHighMassCuts = [&](float leadScore, float subScore, int leadEta, int subEta) {
    return (leadScore > leadCutHighMass1_[leadEta] && subScore > subCutHighMass1_[subEta]) ||
           (leadScore > leadCutHighMass2_[leadEta] && subScore > subCutHighMass2_[subEta]) ||
           (leadScore > leadCutHighMass3_[leadEta] && subScore > subCutHighMass3_[subEta]);
  };

  // Lambda to evaluate a candidate pair
  auto evaluatePair = [&](const edm::Ref<reco::RecoEcalCandidateCollection>& refi,
                          const edm::Ref<reco::RecoEcalCandidateCollection>& refj) {
    float mvaScorei = (*mvaMap).find(refi)->val;
    float mvaScorej = (*mvaMap).find(refj)->val;

    int etai = (std::abs(refi->eta()) < 1.5) ? 0 : 1;
    int etaj = (std::abs(refj->eta()) < 1.5) ? 0 : 1;

    double mass = (refi->p4() + refj->p4()).M();
    if (mass < highMassCut_)
      return false;

    if (mvaScorei >= mvaScorej) {
      return passesHighMassCuts(mvaScorei, mvaScorej, etai, etaj);
    } else {
      return passesHighMassCuts(mvaScorej, mvaScorei, etaj, etai);
    }
  };

  // Loop through candidates
  for (size_t i = 0; i < recCollection->size(); ++i) {
    edm::Ref<reco::RecoEcalCandidateCollection> refi(recCollection, i);
    for (size_t j = i + 1; j < recCollection->size(); ++j) {
      edm::Ref<reco::RecoEcalCandidateCollection> refj(recCollection, j);
      if (evaluatePair(refi, refj)) {
        return true;
      }
    }
  }
  return false;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaDoubleXGBoostCombFilter);
