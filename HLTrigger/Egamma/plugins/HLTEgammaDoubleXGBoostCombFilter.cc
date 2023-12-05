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

  const double lowMassCut_;
  const std::vector<double> leadCutLowMass1_;
  const std::vector<double> subCutLowMass1_;
  const std::vector<double> leadCutLowMass2_;
  const std::vector<double> subCutLowMass2_;
  const std::vector<double> leadCutLowMass3_;
  const std::vector<double> subCutLowMass3_;

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

      lowMassCut_(config.getParameter<double>("lowMassCut")),
      leadCutLowMass1_(config.getParameter<std::vector<double>>("leadCutLowMass1")),
      subCutLowMass1_(config.getParameter<std::vector<double>>("subCutLowMass1")),
      leadCutLowMass2_(config.getParameter<std::vector<double>>("leadCutLowMass2")),
      subCutLowMass2_(config.getParameter<std::vector<double>>("subCutLowMass2")),
      leadCutLowMass3_(config.getParameter<std::vector<double>>("leadCutLowMass3")),
      subCutLowMass3_(config.getParameter<std::vector<double>>("subCutLowMass3")),

      candToken_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("candTag"))),
      mvaToken_(consumes<reco::RecoEcalCandidateIsolationMap>(config.getParameter<edm::InputTag>("mvaPhotonTag"))) {}

void HLTEgammaDoubleXGBoostCombFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);

  desc.add<double>("highMassCut", 95.0);
  desc.add<std::vector<double>>("leadCutHighMass1", {0.98, 0.95});
  desc.add<std::vector<double>>("subCutHighMass1", {0.0, 0.04});
  desc.add<std::vector<double>>("leadCutHighMass2", {0.85, 0.85});
  desc.add<std::vector<double>>("subCutHighMass2", {0.04, 0.08});
  desc.add<std::vector<double>>("leadCutHighMass3", {0.30, 0.50});
  desc.add<std::vector<double>>("subCutHighMass3", {0.15, 0.20});

  desc.add<double>("lowMassCut", 60.0);
  desc.add<std::vector<double>>("leadCutLowMass1", {0.98, 0.90});
  desc.add<std::vector<double>>("subCutLowMass1", {0.04, 0.05});
  desc.add<std::vector<double>>("leadCutLowMass2", {0.90, 0.80});
  desc.add<std::vector<double>>("subCutLowMass2", {0.10, 0.10});
  desc.add<std::vector<double>>("leadCutLowMass3", {0.60, 0.60});
  desc.add<std::vector<double>>("subCutLowMass3", {0.30, 0.30});

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
    int eta1 = (std::abs(EtaSCi) < 1.5) ? 0 : 1;
    float mvaScorei = (*mvaMap).find(refi)->val;
    math::XYZTLorentzVector p4i = refi->p4();
    for (size_t j = i + 1; j < recCollection->size(); j++) {
      edm::Ref<reco::RecoEcalCandidateCollection> refj(recCollection, j);
      float EtaSCj = refj->eta();
      int eta2 = (std::abs(EtaSCj) < 1.5) ? 0 : 1;
      float mvaScorej = (*mvaMap).find(refj)->val;
      math::XYZTLorentzVector p4j = refj->p4();
      math::XYZTLorentzVector pairP4 = p4i + p4j;
      double mass = pairP4.M();
      if (mass >= highMassCut_) {
        if (mvaScorei >= mvaScorej && ((mvaScorei > leadCutHighMass1_[eta1] && mvaScorej > subCutHighMass1_[eta2]) ||
                                       (mvaScorei > leadCutHighMass2_[eta1] && mvaScorej > subCutHighMass2_[eta2]) ||
                                       (mvaScorei > leadCutHighMass3_[eta1] && mvaScorej > subCutHighMass3_[eta2]))) {
          accept = true;
        }  //if scoreI > scoreJ
        else if (mvaScorej > mvaScorei &&
                 ((mvaScorej > leadCutHighMass1_[eta1] && mvaScorei > subCutHighMass1_[eta2]) ||
                  (mvaScorej > leadCutHighMass2_[eta1] && mvaScorei > subCutHighMass2_[eta2]) ||
                  (mvaScorej > leadCutHighMass3_[eta1] && mvaScorei > subCutHighMass3_[eta2]))) {
          accept = true;
        }  // if scoreJ > scoreI
      }    //If high mass
      else if (mass > lowMassCut_ && mass < highMassCut_) {
        if (mvaScorei >= mvaScorej && ((mvaScorei > leadCutLowMass1_[eta1] && mvaScorej > subCutLowMass1_[eta2]) ||
                                       (mvaScorei > leadCutLowMass2_[eta1] && mvaScorej > subCutLowMass2_[eta2]) ||
                                       (mvaScorei > leadCutLowMass3_[eta1] && mvaScorej > subCutLowMass3_[eta2]))) {
          accept = true;
        }  //if scoreI > scoreJ
        else if (mvaScorej > mvaScorei && ((mvaScorej > leadCutLowMass1_[eta1] && mvaScorei > subCutLowMass1_[eta2]) ||
                                           (mvaScorej > leadCutLowMass2_[eta1] && mvaScorei > subCutLowMass2_[eta2]) ||
                                           (mvaScorej > leadCutLowMass3_[eta1] && mvaScorei > subCutLowMass3_[eta2]))) {
          accept = true;
        }  //if scoreJ > scoreI
      }    //If low mass
    }      //j loop
  }        //i loop
  return accept;
}  //Definition

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaDoubleXGBoostCombFilter);
