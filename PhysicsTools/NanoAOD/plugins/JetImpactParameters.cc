/*
////////////////////////////////////////////////

Jet Impact parameter information for displaced tau collection : Pritam Palit, created on 01/09/2025

//////////////////////////////////////////////////
 */

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

void vector_test(std::vector<Float_t>& values) {
  for (auto& value : values) {
    if (std::isnan(value)) {
      throw std::runtime_error("Jet IP output: NaN detected.");
    } else if (std::isinf(value)) {
      throw std::runtime_error("Jet IP output: Infinity detected.");
    } else if (!std::isfinite(value)) {
      throw std::runtime_error("Jet IP output: Non-standard value detected.");
    }
  }
}

class JetImpactParameters : public edm::stream::EDProducer<> {
public:
  explicit JetImpactParameters(const edm::ParameterSet&);
  ~JetImpactParameters() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<pat::JetCollection> jetsToken_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidatesToken_;
  const double deltaRMax_;
};

JetImpactParameters::JetImpactParameters(const edm::ParameterSet& config)
    : jetsToken_(consumes<pat::JetCollection>(config.getParameter<edm::InputTag>("jets"))),
      pfCandidatesToken_(consumes<pat::PackedCandidateCollection>(config.getParameter<edm::InputTag>("pfCandidates"))),
      deltaRMax_(config.getParameter<double>("deltaRMax")) {
  produces<edm::ValueMap<Float_t>>("jetDxy");
  produces<edm::ValueMap<Float_t>>("jetDz");
  produces<edm::ValueMap<Float_t>>("jetDxyError");
  produces<edm::ValueMap<Float_t>>("jetDzError");
  produces<edm::ValueMap<Float_t>>("jetCharge");
}

void JetImpactParameters::produce(edm::Event& event, const edm::EventSetup& setup) {
  // Get jets and PFCandidates
  auto jets = event.getHandle(jetsToken_);

  std::vector<Float_t> v_jetDxy(jets->size(), -1.0);
  std::vector<Float_t> v_jetDz(jets->size(), -1.0);
  std::vector<Float_t> v_jetDxyError(jets->size(), -1.0);
  std::vector<Float_t> v_jetDzError(jets->size(), -1.0);
  std::vector<Float_t> v_jetCharge(jets->size(), -1.0);

  // Loop over jets
  for (size_t jetIndex = 0; jetIndex < jets->size(); ++jetIndex) {
    const auto& jet = jets->at(jetIndex);
    const auto& jetP4 = jet.polarP4();

    // Find the leading charged PFCandidate within deltaR < 0.4
    const pat::PackedCandidate* leadingChargedPFCandidate = nullptr;
    Float_t leadingPt = -1.0;

    // Loop over jet daughters
    const size_t nDaughters = jet.numberOfDaughters();
    for (size_t i = 0; i < nDaughters; ++i) {
      const auto& daughterPtr = jet.daughterPtr(i);
      const auto* daughter = dynamic_cast<const pat::PackedCandidate*>(daughterPtr.get());

      // Skip if not a charged candidate or doesn't have track details
      if (!daughter || daughter->charge() == 0 || !daughter->hasTrackDetails())
        continue;

      Float_t deltaR = reco::deltaR(daughter->polarP4(), jetP4);
      if (deltaR > deltaRMax_)
        continue;

      if (daughter->pt() > leadingPt) {
        leadingPt = daughter->pt();
        leadingChargedPFCandidate = daughter;
      }
    }

    if (leadingChargedPFCandidate) {
      v_jetDxy.at(jetIndex) = leadingChargedPFCandidate->dxy();
      v_jetDz.at(jetIndex) = leadingChargedPFCandidate->dz();
      v_jetDxyError.at(jetIndex) = leadingChargedPFCandidate->dxyError();
      v_jetDzError.at(jetIndex) = leadingChargedPFCandidate->dzError();
      v_jetCharge.at(jetIndex) = leadingChargedPFCandidate->charge();
    }
  }

  vector_test(v_jetDxy);
  vector_test(v_jetDz);
  vector_test(v_jetDxyError);
  vector_test(v_jetDzError);
  vector_test(v_jetCharge);

  auto vm_jetDxy = std::make_unique<edm::ValueMap<Float_t>>();
  edm::ValueMap<Float_t>::Filler filler_jetDxy(*vm_jetDxy);
  filler_jetDxy.insert(jets, v_jetDxy.begin(), v_jetDxy.end());
  filler_jetDxy.fill();
  event.put(std::move(vm_jetDxy), "jetDxy");

  auto vm_jetDz = std::make_unique<edm::ValueMap<Float_t>>();
  edm::ValueMap<Float_t>::Filler filler_jetDz(*vm_jetDz);
  filler_jetDz.insert(jets, v_jetDz.begin(), v_jetDz.end());
  filler_jetDz.fill();
  event.put(std::move(vm_jetDz), "jetDz");

  auto vm_jetDxyError = std::make_unique<edm::ValueMap<Float_t>>();
  edm::ValueMap<Float_t>::Filler filler_jetDxyError(*vm_jetDxyError);
  filler_jetDxyError.insert(jets, v_jetDxyError.begin(), v_jetDxyError.end());
  filler_jetDxyError.fill();
  event.put(std::move(vm_jetDxyError), "jetDxyError");

  auto vm_jetDzError = std::make_unique<edm::ValueMap<Float_t>>();
  edm::ValueMap<Float_t>::Filler filler_jetDzError(*vm_jetDzError);
  filler_jetDzError.insert(jets, v_jetDzError.begin(), v_jetDzError.end());
  filler_jetDzError.fill();
  event.put(std::move(vm_jetDzError), "jetDzError");

  auto vm_jetCharge = std::make_unique<edm::ValueMap<Float_t>>();
  edm::ValueMap<Float_t>::Filler filler_jetCharge(*vm_jetCharge);
  filler_jetCharge.insert(jets, v_jetCharge.begin(), v_jetCharge.end());
  filler_jetCharge.fill();
  event.put(std::move(vm_jetCharge), "jetCharge");
}

DEFINE_FWK_MODULE(JetImpactParameters);
