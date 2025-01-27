#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/MultiJetId.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include <cmath>
#include <vector>

#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

using namespace l1t;

class L1MultiJetProducer : public edm::stream::EDProducer<> {
public:
  explicit L1MultiJetProducer(const edm::ParameterSet&);
  ~L1MultiJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<MultiJetId> fJetId_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  bool const fUseRawPt_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;
  std::vector<l1ct::JetTagClass> classes_;

  hls4mlEmulator::ModelLoader loader;
  std::shared_ptr<hls4mlEmulator::Model> model;
};

L1MultiJetProducer::L1MultiJetProducer(const edm::ParameterSet& cfg)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      fUseRawPt_(cfg.getParameter<bool>("useRawPt")),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      loader(hls4mlEmulator::ModelLoader(cfg.getParameter<string>("MultiJetPath"))) {
  std::vector<std::string> classes = cfg.getParameter<std::vector<std::string>>("classes");
  for(unsigned i = 0; i < classes.size(); i++){
    classes_.push_back(l1ct::JetTagClass(classes[i]));
  }
  model = loader.load_model();
  fJetId_ = std::make_unique<MultiJetId>(model, fNParticles_);
  produces<l1t::PFJetCollection>("L1PFMultiJets");
}

void L1MultiJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);
  std::vector<l1t::PFJet> taggedJets;

  for (const auto& srcjet : *jets) {
    l1ct::Jet ctHWTaggedJet = l1ct::Jet::unpack(srcjet.encodedJet(l1t::PFJet::HWEncoding::CT));
    if (((fUseRawPt_ ? srcjet.rawPt() : srcjet.pt()) < fMinPt_) || std::abs(srcjet.eta()) > fMaxEta_ ||
        taggedJets.size() >= fMaxJets_) {
      ctHWTaggedJet.clear();
      continue;
    }
    std::vector<float> JetScore_float = fJetId_->computeFixed(srcjet, fUseRawPt_);
    for(unsigned i = 0; i < JetScore_float.size(); i++){
      ctHWTaggedJet.hwTagScores[i] = JetScore_float[i];
    }
    l1gt::Jet gtHWTaggedJet = ctHWTaggedJet.toGT();
    // TODO set the regressed pT instead of the srcjet pt
    l1t::PFJet edmTaggedJet(srcjet.pt(), srcjet.eta(), srcjet.phi(), srcjet.mass(),
                            gtHWTaggedJet.v3.pt.V, gtHWTaggedJet.v3.eta.V, gtHWTaggedJet.v3.phi.V
                           );
    edmTaggedJet.setEncodedJet(l1t::PFJet::HWEncoding::CT, ctHWTaggedJet.pack());
    edmTaggedJet.setEncodedJet(l1t::PFJet::HWEncoding::GT, gtHWTaggedJet.pack());
    taggedJets.push_back(edmTaggedJet);
  }
  std::sort(taggedJets.begin(), taggedJets.end(), [](l1t::PFJet a, l1t::PFJet b){ return (a.pt() > b.pt()); });

  std::unique_ptr<l1t::PFJetCollection> taggedJetsCollection(new l1t::PFJetCollection);
  taggedJetsCollection->swap(taggedJets);
  iEvent.put(std::move(taggedJetsCollection), "L1PFMultiJets");
}


void L1MultiJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("scPFL1Puppi"));
  desc.add<bool>("useRawPt", true);
  desc.add<std::string>("MultiJetPath", std::string("L1Trigger/Phase2L1ParticleFlow/data/MultiJetBaseline"));
  desc.add<int>("maxJets", 16);
  desc.add<int>("nParticles", 16);
  desc.add<double>("minPt", 20);
  desc.add<double>("maxEta", 2.4);
  desc.add<edm::InputTag>("vtx", edm::InputTag("L1VertexFinderEmulator", "L1VerticesEmulation"));
  desc.add<std::vector<std::string>>("classes", {"uds", "g", "b", "c", "tau_p", "tau_n", "e", "mu"});
  descriptions.add("L1MultiJetProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1MultiJetProducer);
