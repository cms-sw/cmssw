// system include files
#include <memory>
#include <iostream>

// user include files
#include "PomwigAnalyzer.h"


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

PomwigAnalyzer::PomwigAnalyzer(const edm::ParameterSet& iConfig)
{
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  hist_t = new TH1D("hist_t","t proton",100,-1.4,0);
  hist_xigen = new TH1D("hist_xigen","#xi proton",100,0.,0.1);
}


PomwigAnalyzer::~PomwigAnalyzer()
{
 
}

// ------------ method called to for each event  ------------
void
PomwigAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
     if(((*it)->pdg_id() == 2212)&&((*it)->status() == 1)&&((*it)->momentum().pz() > 5200.)){
	proton1 = *it;
     } else if(((*it)->pdg_id() == 2212)&&((*it)->status() == 1)&&((*it)->momentum().pz() < -5200.)){
     	proton2 = *it;
     }		
   }
   if(proton1){
		std::cout << "Proton 1: " << proton1->momentum().perp() << "  " << proton1->momentum().pseudoRapidity() << "  " << proton1->momentum().phi() << std::endl;
		HepLorentzVector proton1in(0.,0.,7000.,7000.);
   		double t1 = (proton1->momentum() - proton1in).m2();
   		double xigen1 = 1 - proton1->momentum().pz()/7000.;
		hist_t->Fill(t1);
		hist_xigen->Fill(xigen1);
   }	

   if(proton2){
	std::cout << "Proton 2: " << proton2->momentum().perp() << "  " << proton2->momentum().pseudoRapidity() << "  " << proton2->momentum().phi() << std::endl;	
	HepLorentzVector proton2in(0.,0.,-7000.,7000.);
   	double t2 = (proton2->momentum() - proton2in).m2();
   	double xigen2 = 1 + proton2->momentum().pz()/7000.;
	hist_t->Fill(t2);
        hist_xigen->Fill(xigen2);
   }

}
// ------------ method called once each job just before starting event loop  ------------
void 
PomwigAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PomwigAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  hist_t->Write();
  hist_xigen->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(PomwigAnalyzer);
