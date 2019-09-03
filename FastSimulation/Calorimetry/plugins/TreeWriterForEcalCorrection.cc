// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"

class TreeWriterForEcalCorrection : public edm::EDAnalyzer {
public:
  explicit TreeWriterForEcalCorrection(const edm::ParameterSet&);
  ~TreeWriterForEcalCorrection() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::Service<TFileService> file;
  TTree* tree;
  float tree_e, tree_eta, tree_response;
};

TreeWriterForEcalCorrection::TreeWriterForEcalCorrection(const edm::ParameterSet& iConfig) {
  tree = file->make<TTree>("responseTree", "same info as 3dhisto");
  tree->Branch("e", &tree_e, "e/F");
  tree->Branch("eta", &tree_eta, "eta/F");
  tree->Branch("r", &tree_response, "r/F");
}

void TreeWriterForEcalCorrection::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get generated particles
  edm::Handle<reco::GenParticleCollection> GenParticles;
  iEvent.getByLabel("genParticles", "", GenParticles);

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
  edm::Handle<edm::PCaloHitContainer> SimHitsEB;
  edm::Handle<edm::PCaloHitContainer> SimHitsEE;
  edm::Handle<edm::PCaloHitContainer> SimHitsES;

  // Finds out automatically, if this is fullsim or fastsim
  bool isFastSim = iEvent.getByLabel("fastSimProducer", "EcalHitsEB", SimHitsEB);
  if (isFastSim) {
    iEvent.getByLabel("fastSimProducer", "EcalHitsEE", SimHitsEE);
    iEvent.getByLabel("fastSimProducer", "EcalHitsES", SimHitsES);
  } else {
    iEvent.getByLabel("g4SimHits", "EcalHitsEB", SimHitsEB);
    iEvent.getByLabel("g4SimHits", "EcalHitsEE", SimHitsEE);
    iEvent.getByLabel("g4SimHits", "EcalHitsES", SimHitsES);
  }

  // merge them into one single vector
  auto SimHits = *SimHitsEB;
  SimHits.insert(SimHits.end(), SimHitsEE->begin(), SimHitsEE->end());
  SimHits.insert(SimHits.end(), SimHitsES->begin(), SimHitsES->end());

  // As we only had one generated particle (and hopefully no pileup),
  // the total energy is due to the generated particle only
  float energyTotal = 0;
  for (auto const& Hit : SimHits) {
    energyTotal += Hit.energy();
  }

  tree_e = genE;
  tree_eta = genEta;
  tree_response = energyTotal / genE;
  tree->Fill();
}

void TreeWriterForEcalCorrection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("ecalScaleFactorCalculator", desc);
}

DEFINE_FWK_MODULE(TreeWriterForEcalCorrection);
