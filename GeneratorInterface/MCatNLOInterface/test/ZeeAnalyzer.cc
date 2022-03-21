/* This is en example for an Analyzer of a MCatNLO HeoMCProduct
   The analyzer fills a histogram with the event weights,
   and looks for electron/positron pairs and fills a histogram
   with the invaraint mass of the two. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
//
//

// system include files
#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HepMC/WeightContainer.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TH1D.h"
#include "TFile.h"

//
// class decleration
//

class ZeeAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ZeeAnalyzer(const edm::ParameterSet&);
  ~ZeeAnalyzer() override = default;

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::InputTag hepMCProductTag_;
  const edm::InputTag genEventInfoProductTag_;
  const std::string outputFilename;

  const edm::EDGetTokenT<edm::HepMCProduct> tokHepMC_;
  const edm::EDGetTokenT<GenEventInfoProduct> tokEvent_;
  TH1D* weight_histo;
  TH1D* invmass_histo;
  TH1D* Zpt;
  TH1D* hardpt;
  TH1D* softpt;
  TH1D* hardeta;
  TH1D* softeta;
  TH1D* hardphi;
  TH1D* softphi;

  double sumWeights;
};

ZeeAnalyzer::ZeeAnalyzer(const edm::ParameterSet& iConfig)
    : hepMCProductTag_(iConfig.getParameter<edm::InputTag>("hepMCProductTag")),
      genEventInfoProductTag_(iConfig.getParameter<edm::InputTag>("genEventInfoProductTag")),
      outputFilename(iConfig.getUntrackedParameter<std::string>("OutputFilename", "dummy.root")),
      tokHepMC_(consumes<edm::HepMCProduct>(hepMCProductTag_)),
      tokEvent_(consumes<GenEventInfoProduct>(genEventInfoProductTag_)) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;

  sumWeights = 0.0;

  weight_histo = fs->make<TH1D>("weight_histo", "weight_histo", 20, -10, 10);
  invmass_histo = fs->make<TH1D>("invmass_histo", "invmass_histo", 160, 70, 110);
  Zpt = fs->make<TH1D>("Zpt", "Zpt", 200, 0, 200);
  hardpt = fs->make<TH1D>("hardpt", "hardpt", 200, 0, 200);
  softpt = fs->make<TH1D>("softpt", "softpt", 200, 0, 200);
  hardeta = fs->make<TH1D>("hardeta", "hardeta", 200, -5, 5);
  softeta = fs->make<TH1D>("softeta", "softeta", 200, -5, 5);
}

// ------------ method called to for each event  ------------
void ZeeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get HepMC::GenEvent ...
  const edm::Handle<edm::HepMCProduct>& evt_h = iEvent.getHandle(tokHepMC_);
  HepMC::GenEvent* evt = new HepMC::GenEvent(*(evt_h->GetEvent()));

  const edm::Handle<GenEventInfoProduct>& evt_info = iEvent.getHandle(tokEvent_);

  double weight = evt_info->weight();
  if (weight)
    weight_histo->Fill(weight);

  // look for stable electrons/positrons
  std::vector<HepMC::GenParticle*> electrons;
  for (HepMC::GenEvent::particle_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
    if (abs((*it)->pdg_id()) == 11 && (*it)->status() == 1)
      electrons.push_back(*it);
  }

  // if there are at least two electrons/positrons,
  // calculate invarant mass of first two and fill it into histogram

  double inv_mass = 0.0;
  double Zpt_ = 0.0;
  if (electrons.size() > 1) {
    math::XYZTLorentzVector tot_momentum(electrons[0]->momentum());
    math::XYZTLorentzVector mom2(electrons[1]->momentum());
    tot_momentum += mom2;
    inv_mass = sqrt(tot_momentum.mass2());
    Zpt_ = tot_momentum.pt();

    // IMPORTANT: use the weight of the event ...

    double weight_sign = (weight > 0) ? 1. : -1.;
    invmass_histo->Fill(inv_mass, weight_sign);
    Zpt->Fill(Zpt_, weight_sign);
    if (electrons[0]->momentum().perp() > electrons[1]->momentum().perp()) {
      hardpt->Fill(electrons[0]->momentum().perp(), weight_sign);
      softpt->Fill(electrons[1]->momentum().perp(), weight_sign);
      hardeta->Fill(electrons[0]->momentum().eta(), weight_sign);
      softeta->Fill(electrons[1]->momentum().eta(), weight_sign);
    } else {
      hardpt->Fill(electrons[1]->momentum().perp(), weight_sign);
      softpt->Fill(electrons[0]->momentum().perp(), weight_sign);
      hardeta->Fill(electrons[1]->momentum().eta(), weight_sign);
      softeta->Fill(electrons[0]->momentum().eta(), weight_sign);
    }

    sumWeights += weight_sign;
  }

  delete evt;
}

// ------------ method called once each job just after ending the event loop  ------------
void ZeeAnalyzer::endJob() { std::cout << " total sum wieghts = " << sumWeights << std::endl; }

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeAnalyzer);
