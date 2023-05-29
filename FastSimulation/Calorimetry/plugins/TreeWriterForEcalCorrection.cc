// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"

class TreeWriterForEcalCorrection : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TreeWriterForEcalCorrection(const edm::ParameterSet&);
  ~TreeWriterForEcalCorrection() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<reco::GenParticleCollectio> tok_gen_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_ebf_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_eef_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_esf_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_ebs_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_ees_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_ess_;
  TTree* tree_;
  float tree_e_, tree_eta_, tree_response_;
};

TreeWriterForEcalCorrection::TreeWriterForEcalCorrection(const edm::ParameterSet& iConfig)
    : tok_gen_(consumes<reco::GenParticleCollectio>(edm::InputTag("genParticles", ""))),
      tok_ebf_(consumes<edm::PCaloHitContainer>(edm::InputTag("fastSimProducer", "EcalHitsEB"))),
      tok_eef_(consumes<edm::PCaloHitContainer>(edm::InputTag("fastSimProducer", "EcalHitsEE"))),
      tok_esf_(consumes<edm::PCaloHitContainer>(edm::InputTag("fastSimProducer", "EcalHitsES"))),
      tok_ebs_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"))),
      tok_ees_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEE"))),
      tok_ess_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsES"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> file;
  tree_ = file->make<TTree>("responseTree", "same info as 3dhisto");
  tree_->Branch("e", &tree_e_, "e/F");
  tree_->Branch("eta", &tree_eta_, "eta/F");
  tree_->Branch("r", &tree_response_, "r/F");
}

void TreeWriterForEcalCorrection::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get generated particles
  const edm::Handle<reco::GenParticleCollection>& GenParticles = iEvent.getHandle(tok_gen_);

  // As this module is intended for single particle guns, there should be
  // exactly one generated particle.
  if (GenParticles->size() != 1) {
    throw cms::Exception("MismatchedInputFiles") << "Intended for particle guns only\n";
  }

  // I assume here that the tracker simulation is disabled and no vertex
  // smearing is done, so that the generated particles is exactly the same
  // particle as the particle hitting the entrace of the ECAL.
  reco::GenParticle gen = GenParticles->at(0);
  float genE = gen.energy();
  float genEta = gen.eta();
  // genEta is positive for my photongun sample per definition.
  // If not, you could take the absolute value here.

  // get sim hits
  edm::Handle<edm::PCaloHitContainer> SimHitsEB = iEvent.getHandle(tok_ebf_);
  // Finds out automatically, if this is fullsim or fastsim
  bool isFastSim = SimHitsEB.isValid();
  if (!isFastSim)
    SimHitsEB = iEvent.getHandle(tok_ebs_);
  const edm::Handle<edm::PCaloHitContainer> SimHitsEE =
      isFastSim ? iEvent.getHandle(tok_eef_) : iEvent.getHandle(tok_ees_);
  const edm::Handle<edm::PCaloHitContainer> SimHitsES =
      isFastSim ? iEvent.getHandle(tok_esf_) : iEvent.getHandle(tok_ess_);

  // merge them into one single vector
  auto SimHits = *(SimHitsEB.product());
  SimHits.insert(SimHits.end(), SimHitsEE->begin(), SimHitsEE->end());
  SimHits.insert(SimHits.end(), SimHitsES->begin(), SimHitsES->end());

  // As we only had one generated particle (and hopefully no pileup),
  // the total energy is due to the generated particle only
  float energyTotal = 0;
  for (auto const& Hit : SimHits) {
    energyTotal += Hit.energy();
  }

  tree_e_ = genE;
  tree_eta_ = genEta;
  tree_response_ = energyTotal / genE;
  tree_->Fill();
}

void TreeWriterForEcalCorrection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("ecalScaleFactorCalculator", desc);
}

DEFINE_FWK_MODULE(TreeWriterForEcalCorrection);
