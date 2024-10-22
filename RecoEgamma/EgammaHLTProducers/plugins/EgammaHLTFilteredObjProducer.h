#ifndef RECOEGAMMA_EGAMMAHLTPRODUCERS_EGAMMAHLTFILTEREDOBJPRODUCER_H
#define RECOEGAMMA_EGAMMAHLTPRODUCERS_EGAMMAHLTFILTEREDOBJPRODUCER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

namespace {
  template <typename T>
  edm::Handle<T> getHandle(const edm::Event& event, const edm::EDGetTokenT<T>& token) {
    edm::Handle<T> handle;
    event.getByToken(token, handle);
    return handle;
  }

}  // namespace

template <typename OutCollType>
class EgammaHLTFilteredObjProducer : public edm::stream::EDProducer<> {
public:
  class SelectionCut {
  public:
    SelectionCut(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
        : ebCut_(pset.getParameter<edm::ParameterSet>("barrelCut")),
          eeCut_(pset.getParameter<edm::ParameterSet>("endcapCut")),
          varToken_(iC.consumes<reco::RecoEcalCandidateIsolationMap>(pset.getParameter<edm::InputTag>("var"))) {}

    ~SelectionCut() = default;

    bool operator()(const reco::RecoEcalCandidateRef& cand) const {
      CutValues cut = std::abs(cand->eta()) < kEBEEEtaBound_ ? ebCut_ : eeCut_;
      return cut(*cand, getVar(cand));
    }

    float getVar(const reco::RecoEcalCandidateRef& cand) const {
      auto res = varHandle_->find(cand);
      if (res != varHandle_->end())
        return res->val;
      else {
        //FIX ME: add some provenance info to this
        throw cms::Exception("LogicError") << " candidate not found in collection ";
      }
    }

    void getHandles(const edm::Event& event) { event.getByToken(varToken_, varHandle_); }

  private:
    struct CutValues {
      float cut;
      float cutOverE;
      float cutOverE2;
      bool useEt;
      bool doAnd;
      std::function<bool(float, float)> compFunc;

      CutValues(const edm::ParameterSet& pset)
          : cut(pset.getParameter<double>("cut")),
            cutOverE(pset.getParameter<double>("cutOverE")),
            cutOverE2(pset.getParameter<double>("cutOverE2")),
            useEt(pset.getParameter<bool>("useEt")),
            doAnd(pset.getParameter<bool>("doAnd")),
            compFunc(std::less<float>()) {}

      bool operator()(const reco::RecoEcalCandidate& cand, float value) const {
        if (!doAnd) {
          if (compFunc(value, cut))
            return true;
          else {
            float energyInv = useEt ? 1. / cand.et() : 1. / cand.energy();
            return compFunc(value * energyInv, cutOverE) || compFunc(value * energyInv * energyInv, cutOverE2);
          }
        } else {
          float energy = useEt ? cand.et() : cand.energy();
          return compFunc(value, cut + cutOverE * energy + cutOverE2 * energy * energy);
        }
      }
    };

    CutValues ebCut_;
    CutValues eeCut_;
    edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> varToken_;
    edm::Handle<reco::RecoEcalCandidateIsolationMap> varHandle_;
  };

  explicit EgammaHLTFilteredObjProducer(const edm::ParameterSet& pset);
  ~EgammaHLTFilteredObjProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //different collection types may need to specialise this function
  //eg if you wanted superclusters it would output.push_back(cand->superCluster())
  static void addObj(const reco::RecoEcalCandidateRef& cand, OutCollType& output) { output.push_back(cand); }

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candsToken_;
  std::vector<SelectionCut> cuts_;
  float minEtCutEB_;
  float minEtCutEE_;
  static constexpr float kEBEEEtaBound_ = 1.479;
};

template <typename OutCollType>
EgammaHLTFilteredObjProducer<OutCollType>::EgammaHLTFilteredObjProducer(const edm::ParameterSet& pset)
    : candsToken_(consumes<reco::RecoEcalCandidateCollection>(pset.getParameter<edm::InputTag>("cands"))),
      minEtCutEB_(pset.getParameter<double>("minEtCutEB")),
      minEtCutEE_(pset.getParameter<double>("minEtCutEE")) {
  const auto& cutPsets = pset.getParameter<std::vector<edm::ParameterSet> >("cuts");
  for (auto& cutPset : cutPsets) {
    cuts_.push_back(SelectionCut(cutPset, consumesCollector()));
  }

  produces<OutCollType>();
}

template <typename OutCollType>
void EgammaHLTFilteredObjProducer<OutCollType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("cands", edm::InputTag("hltEgammaCandidates"));
  desc.add<double>("minEtCutEB", 0);
  desc.add<double>("minEtCutEE", 0);

  edm::ParameterSetDescription cutsDesc;
  edm::ParameterSetDescription regionCutsDesc;
  regionCutsDesc.add<double>("cut", -1);
  regionCutsDesc.add<double>("cutOverE", -1);
  regionCutsDesc.add<double>("cutOverE2", -1);
  regionCutsDesc.add<bool>("useEt", false);
  regionCutsDesc.add<bool>("doAnd", false);
  edm::ParameterSet cutDefaults;
  cutDefaults.addParameter<double>("cutOverE", 0.2);
  cutDefaults.addParameter<double>("useEt", false);
  cutDefaults.addParameter<double>("doAnd", false);

  cutsDesc.add<edm::ParameterSetDescription>("barrelCut", regionCutsDesc);
  cutsDesc.add<edm::ParameterSetDescription>("endcapCut", regionCutsDesc);
  cutsDesc.add<edm::InputTag>("var", edm::InputTag("hltEgammaHoverE"));

  edm::ParameterSet defaults;
  defaults.addParameter<edm::InputTag>("var", edm::InputTag("hltEgammaHoverE"));
  defaults.addParameter<edm::ParameterSet>("barrelCut", cutDefaults);
  defaults.addParameter<edm::ParameterSet>("endcapCut", cutDefaults);
  desc.addVPSet("cuts", cutsDesc, std::vector<edm::ParameterSet>{defaults});
  descriptions.addWithDefaultLabel(desc);
}

template <typename OutCollType>
void EgammaHLTFilteredObjProducer<OutCollType>::produce(edm::Event& iEvent, const edm::EventSetup&) {
  for (auto& cut : cuts_)
    cut.getHandles(iEvent);
  auto candsHandle = getHandle(iEvent, candsToken_);

  auto output = std::make_unique<OutCollType>();

  for (size_t candNr = 0; candNr < candsHandle->size(); candNr++) {
    reco::RecoEcalCandidateRef candRef(candsHandle, candNr);
    bool passAllCuts = true;

    float minEtCut = std::abs(candRef->eta()) < kEBEEEtaBound_ ? minEtCutEB_ : minEtCutEE_;
    if (candRef->et() < minEtCut)
      passAllCuts = false;

    if (passAllCuts) {
      for (const auto& cut : cuts_) {
        if (!cut(candRef)) {
          passAllCuts = false;
          break;
        }
      }
    }
    if (passAllCuts)
      addObj(candRef, *output);
  }

  iEvent.put(std::move(output));
}

#endif
