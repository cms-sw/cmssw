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
  //producer collection (hltEgammaCandidates(Unseeded))
  const auto& recCollection = event.getHandle(candToken_);

  //get hold of photon MVA association map
  const auto& mvaMap = event.getHandle(mvaToken_);

  std::vector<math::XYZTLorentzVector> p4s(recCollection->size());
  std::vector<bool> isTight(recCollection->size());

  bool accept = false;

  for (size_t i = 0; i < recCollection->size(); i++) {
    edm::Ref<reco::RecoEcalCandidateCollection> refi(recCollection, i);
    float EtaSCi = refi->eta();
    int etai = (std::abs(EtaSCi) < 1.5) ? 0 : 1;
    float mvaScorei = (*mvaMap).find(refi)->val;
    math::XYZTLorentzVector p4i = refi->p4();
    for (size_t j = i + 1; j < recCollection->size(); j++) {
      edm::Ref<reco::RecoEcalCandidateCollection> refj(recCollection, j);
      float EtaSCj = refj->eta();
      int etaj = (std::abs(EtaSCj) < 1.5) ? 0 : 1;
      float mvaScorej = (*mvaMap).find(refj)->val;
      math::XYZTLorentzVector p4j = refj->p4();
      math::XYZTLorentzVector pairP4 = p4i + p4j;
      double mass = pairP4.M();
      if (mass >= highMassCut_) {
        if (mvaScorei >= mvaScorej && ((mvaScorei > leadCutHighMass1_[etai] && mvaScorej > subCutHighMass1_[etaj]) ||
                                       (mvaScorei > leadCutHighMass2_[etai] && mvaScorej > subCutHighMass2_[etaj]) ||
                                       (mvaScorei > leadCutHighMass3_[etai] && mvaScorej > subCutHighMass3_[etaj]))) {
          accept = true;
        }  //if scoreI > scoreJ
        else if (mvaScorej > mvaScorei &&
                 ((mvaScorej > leadCutHighMass1_[etaj] && mvaScorei > subCutHighMass1_[etai]) ||
                  (mvaScorej > leadCutHighMass2_[etaj] && mvaScorei > subCutHighMass2_[etai]) ||
                  (mvaScorej > leadCutHighMass3_[etaj] && mvaScorei > subCutHighMass3_[etai]))) {
          accept = true;
        }  // if scoreJ > scoreI
      }  //If high mass
    }  //j loop
  }  //i loop
  return accept;
}  //Definition

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaDoubleXGBoostCombFilter);
