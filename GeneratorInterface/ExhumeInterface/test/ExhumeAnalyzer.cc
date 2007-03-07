// system include files
#include <memory>
#include <iostream>

// user include files
#include "ExhumeAnalyzer.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenParticle.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TFile.h"

ExhumeAnalyzer::ExhumeAnalyzer(const edm::ParameterSet& iConfig)
{
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  hist_eta = new TH1D("hist_eta","#eta system",100,-4.5,4.5);
  hist_t1 = new TH1D("hist_t1","t proton 1",100,-1.4,0);
  hist_xigen1 = new TH1D("hist_xigen1","#xi proton 1",100,0.,0.1);
  hist_t2 = new TH1D("hist_t1","t proton 1",100,-1.4,0);
  hist_xigen2 = new TH1D("hist_xigen2","#xi proton 2",100,0.,0.1);
}


ExhumeAnalyzer::~ExhumeAnalyzer()
{
 
}

// ------------ method called to for each event  ------------
void
ExhumeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
   // get HepMC::GenEvent ...
   Handle<HepMCProduct> evt_h;
   iEvent.getByType(evt_h);
   HepMC::GenEvent * evt = new  HepMC::GenEvent(*(evt_h->GetEvent()));


   // look for protons
   //std::vector<HepMC::GenParticle*> protons;
   HepMC::GenParticle* proton1 = 0;
   HepMC::GenParticle* proton2 = 0;	
   for(HepMC::GenEvent::particle_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
     if(((*it)->pdg_id() == 2212)&&((*it)->status() == 1)&&((*it)->mother() == 0)&&((*it)->momentum().pz() > 0.)){
	proton1 = *it;
     } else if(((*it)->pdg_id() == 2212)&&((*it)->status() == 1)&&((*it)->mother() == 0)&&((*it)->momentum().pz() < 0.)){
     	proton2 = *it;
     }		
   }
   std::cout << "Proton 1: " << proton1->momentum().perp() << "  " << proton1->momentum().pseudoRapidity() << "  " << proton1->momentum().phi() << std::endl;
   std::cout << "Proton 2: " << proton2->momentum().perp() << "  " << proton2->momentum().pseudoRapidity() << "  " << proton2->momentum().phi() << std::endl;	

   HepLorentzVector proton1in(0.,0.,7000.,7000.);   
   double t1 = (proton1->momentum() - proton1in).m2();
   double xigen1 = 1 - proton1->momentum().pz()/7000.;
   HepLorentzVector proton2in(0.,0.,-7000.,7000.);
   double t2 = (proton2->momentum() - proton2in).m2();
   double xigen2 = 1 + proton2->momentum().pz()/7000.;
   
   //Fill histograms
   hist_t1->Fill(t1);
   hist_xigen1->Fill(xigen1);
   hist_t2->Fill(t2);
   hist_xigen2->Fill(xigen2);	
}
// ------------ method called once each job just before starting event loop  ------------
void 
ExhumeAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ExhumeAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  hist_eta->Write();
  hist_t1->Write();
  hist_xigen1->Write();
  hist_t2->Write();
  hist_xigen2->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(ExhumeAnalyzer);
