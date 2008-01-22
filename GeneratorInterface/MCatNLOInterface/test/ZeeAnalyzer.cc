/* This is en example for an Analyzer of a MCatNLO HeoMCProduct
   The analyzer fills a histogram with the event weights,
   and looks for electron/positron pairs and fills a histogram
   with the invaraint mass of the two. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: ZeeAnalyzer.cc,v 1.6 2007/03/21 15:08:40 fabstoec Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "ZeeAnalyzer.h"











ZeeAnalyzer::ZeeAnalyzer(const edm::ParameterSet& iConfig)
{

  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");


  weight_histo  = new TH1D("weight_histo","weight_histo",20,-10,10);
  invmass_histo = new TH1D("invmass_histo","invmass_histo",40,70,110);
}


ZeeAnalyzer::~ZeeAnalyzer()
{
 
}


// ------------ method called to for each event  ------------
void ZeeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace edm;
  

   // get HepMC::GenEvent ...
   Handle<HepMCProduct> evt_h;
   iEvent.getByType(evt_h);
   HepMC::GenEvent* evt = new  HepMC::GenEvent(*(evt_h->GetEvent()));


   // get weight and fill it to histogram
   HepMC::WeightContainer weights = evt->weights();   
   double weight = weights.front();
   if(weight) weight_histo->Fill(weight);
   
   // look for stable electrons/positrons
   std::vector<HepMC::GenParticle*> electrons;   
   for(HepMC::GenEvent::particle_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
     if(abs((*it)->pdg_id())==11 && (*it)->status()==1)
       electrons.push_back(*it);
   }
   
   // if there are at least two electrons/positrons, 
   // calculate invarant mass of first two and fill it into histogram

   double inv_mass = 0.0;
   if(electrons.size()>1) {
     math::XYZTLorentzVector tot_momentum(electrons[0]->momentum());
     math::XYZTLorentzVector mom2(electrons[1]->momentum());
     tot_momentum += mom2;
     inv_mass = sqrt(tot_momentum.mass2());
   }
   
   // IMPORTANT: use the weight of the event ... 
   double weight_sign = (weight > 0) ? 1. : -1.;
   invmass_histo->Fill(inv_mass,weight_sign);

   delete evt;
}


// ------------ method called once each job just before starting event loop  ------------
void 
ZeeAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZeeAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  weight_histo->Write();
  invmass_histo->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeAnalyzer);
