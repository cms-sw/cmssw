/* This is en example for an Analyzer of a Herwig HeoMCProduct
   and looks for muons pairs and fills a histogram
   with the invaraint mass of the four. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: H4muAnalyzer.cc,v 1.1 2006/11/14 15:15:14 fabstoec Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "H4muAnalyzer.h"


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

H4muAnalyzer::H4muAnalyzer(const edm::ParameterSet& iConfig)
{
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  invmass_histo = new TH1D("invmass_histo","invmass_histo",20,180,200);
}


H4muAnalyzer::~H4muAnalyzer()
{
 
}

// ------------ method called to for each event  ------------
void
H4muAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
   // get HepMC::GenEvent ...
   Handle<HepMCProduct> evt_h;
   iEvent.getByType(evt_h);
   HepMC::GenEvent * evt = new  HepMC::GenEvent(*(evt_h->GetEvent()));


   // look for stable muons
   std::vector<HepMC::GenParticle*> muons;   
   for(HepMC::GenEvent::particle_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
     if(abs((*it)->pdg_id())==13 && (*it)->status()==1)
       muons.push_back(*it);
   }
   
   // if there are at least four muons
   // calculate invarant mass of first two and fill it into histogram
   HepLorentzVector tot_momentum;
   double inv_mass = 0.0;
   std::cout<<muons.size()<<std::endl;
   if(muons.size()>3) {
     tot_momentum = muons[0]->momentum();
     tot_momentum += muons[1]->momentum();
     tot_momentum += muons[2]->momentum();
     tot_momentum += muons[3]->momentum();     
     inv_mass = sqrt(tot_momentum.m2());
   }
   
   invmass_histo->Fill(inv_mass);
   std::cout<<inv_mass<<std::endl;

}


// ------------ method called once each job just before starting event loop  ------------
void 
H4muAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
H4muAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  invmass_histo->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(H4muAnalyzer)
