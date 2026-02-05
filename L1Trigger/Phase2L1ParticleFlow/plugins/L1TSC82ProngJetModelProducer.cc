#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC82ProngJetID.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include <cmath>
#include <vector>

#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

using namespace l1t;

class L1TSC82ProngJetProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TSC82ProngJetProducer(const edm::ParameterSet&);
  ~L1TSC82ProngJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<L1TSC82ProngJetID> fJetId_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;

  hls4mlEmulator::ModelLoader loader;
  std::shared_ptr<hls4mlEmulator::Model> model;
};

L1TSC82ProngJetProducer::L1TSC82ProngJetProducer(const edm::ParameterSet& cfg)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      loader(hls4mlEmulator::ModelLoader(cfg.getParameter<string>("l1tSC82ProngJetModelPath"))) {
  model = loader.load_model();
  fJetId_ = std::make_unique<L1TSC82ProngJetID>(model, fNParticles_);
  produces<l1t::PFJetCollection>("l1tSC82ProngJets");
}

void L1TSC82ProngJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);
  std::vector<l1t::PFJet> taggedJets;

  for (const auto& srcjet : *jets) {
    l1ct::Jet ctHWJet = l1ct::Jet::unpack(srcjet.encodedJet(l1t::PFJet::HWEncoding::CT));

    if (srcjet.pt() < fMinPt_ || std::abs(srcjet.eta()) > fMaxEta_ || taggedJets.size() >= fMaxJets_) {
      ctHWJet.clear();
      continue;
    }
    std::vector<float> JetProngScore_float = fJetId_->computeFixed(srcjet);
    l1gt::WideJet gtwHWJet = l1gt::WideJet::unpack(srcjet.getHWJetGTWide());

    l1t::PFJet edmJet(
        srcjet.pt(), srcjet.eta(), srcjet.phi(), srcjet.mass(), gtwHWJet.v3.pt.V, gtwHWJet.v3.eta.V, gtwHWJet.v3.phi.V);

    std::vector<l1ct::JetTagClass> classes{l1ct::JetTagClass(l1ct::JetTagClass::JetTagClassValue::nprong)};

    edmJet.addTagScores(JetProngScore_float, classes, 1.);
    edmJet.setEncodedJet(l1t::PFJet::HWEncoding::CT, ctHWJet.pack());
    edmJet.setEncodedJet(l1t::PFJet::HWEncoding::GTWide, gtwHWJet.pack());

    std::vector<edm::Ptr<l1t::PFCandidate>> constituents;
    std::for_each(srcjet.constituents().begin(), srcjet.constituents().end(), [&](auto constituent) {
      edmJet.addConstituent(constituent);
    });

    taggedJets.push_back(edmJet);
  }
  std::sort(taggedJets.begin(), taggedJets.end(), [](l1t::PFJet a, l1t::PFJet b) { return (a.pt() > b.pt()); });

  std::unique_ptr<l1t::PFJetCollection> taggedJetsCollection(new l1t::PFJetCollection);
  taggedJetsCollection->swap(taggedJets);
  iEvent.put(std::move(taggedJetsCollection), "l1tSC82ProngJets");
}

void L1TSC82ProngJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("l1tSC8PFL1PuppiEmulator"));
  desc.add<std::string>("l1tSC82ProngJetModelPath", std::string("L1TSC82ProngJetModel_v0"));
  desc.add<int>("maxJets", 16);
  desc.add<int>("nParticles", 8);
  desc.add<double>("minPt", 10);
  desc.add<double>("maxEta", 2.4);
  descriptions.add("l1tSC82ProngJetProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TSC82ProngJetProducer);
