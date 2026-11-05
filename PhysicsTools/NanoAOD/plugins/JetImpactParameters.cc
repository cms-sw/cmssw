/*
////////////////////////////////////////////////

Jet Impact parameter information for displaced tau collection : Pritam Palit, created on 01/09/2025

//////////////////////////////////////////////////
 */

#include <cmath>
#include <memory>
#include <vector>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace {

  constexpr Float_t kUnavailable = -9999.0;

  void vector_test(std::vector<Float_t>& values, const char* label, const edm::EventID& id) {
    for (size_t i = 0; i < values.size(); ++i) {
      if (std::isfinite(values[i]))
        continue;
      edm::LogWarning("JetImpactParameters")
          << "run " << id.run() << " lumi " << id.luminosityBlock() << " event " << id.event() << ": non-finite "
          << label << " (" << values[i] << ") for jet " << i << ", writing " << kUnavailable << " instead.";
      values[i] = kUnavailable;
    }
  }

  void putValueMap(edm::Event& event,
                   const edm::Handle<pat::JetCollection>& jets,
                   const std::vector<Float_t>& values,
                   const std::string& name) {
    auto valueMap = std::make_unique<edm::ValueMap<Float_t>>();
    edm::ValueMap<Float_t>::Filler filler(*valueMap);
    filler.insert(jets, values.begin(), values.end());
    filler.fill();
    event.put(std::move(valueMap), name);
  }

}  // namespace

class JetImpactParameters : public edm::stream::EDProducer<> {
public:
  explicit JetImpactParameters(const edm::ParameterSet&);
  ~JetImpactParameters() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<pat::JetCollection> jetsToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidatesToken_;
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

void JetImpactParameters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("linkedObjectsCHS", "jets"));
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<double>("deltaRMax", 0.4);
  descriptions.addWithDefaultLabel(desc);
}

void JetImpactParameters::produce(edm::Event& event, const edm::EventSetup&) {
  auto jets = event.getHandle(jetsToken_);

  std::vector<Float_t> v_jetDxy(jets->size(), kUnavailable);
  std::vector<Float_t> v_jetDz(jets->size(), kUnavailable);
  std::vector<Float_t> v_jetDxyError(jets->size(), kUnavailable);
  std::vector<Float_t> v_jetDzError(jets->size(), kUnavailable);
  std::vector<Float_t> v_jetCharge(jets->size(), kUnavailable);

  // Loop over jets
  for (size_t jetIndex = 0; jetIndex < jets->size(); ++jetIndex) {
    const auto& jet = jets->at(jetIndex);
    const auto& jetP4 = jet.polarP4();

    // Find the leading charged PFCandidate within deltaR < deltaRMax
    const pat::PackedCandidate* leadingChargedPFCandidate = nullptr;
    Float_t leadingPt = -1.0;

    // Loop over jet daughters
    const size_t nDaughters = jet.numberOfDaughters();
    for (size_t i = 0; i < nDaughters; ++i) {
      const auto& daughterPtr = jet.daughterPtr(i);
      const auto* daughter = dynamic_cast<const pat::PackedCandidate*>(daughterPtr.get());

      // Skip if not a charged candidate or does not have track details
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

  vector_test(v_jetDxy, "jetDxy", event.id());
  vector_test(v_jetDz, "jetDz", event.id());
  vector_test(v_jetDxyError, "jetDxyError", event.id());
  vector_test(v_jetDzError, "jetDzError", event.id());
  vector_test(v_jetCharge, "jetCharge", event.id());

  putValueMap(event, jets, v_jetDxy, "jetDxy");
  putValueMap(event, jets, v_jetDz, "jetDz");
  putValueMap(event, jets, v_jetDxyError, "jetDxyError");
  putValueMap(event, jets, v_jetDzError, "jetDzError");
  putValueMap(event, jets, v_jetCharge, "jetCharge");
}

DEFINE_FWK_MODULE(JetImpactParameters);
