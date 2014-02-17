/* This is en example for an Analyzer of a MCatNLO HeoMCProduct
   The analyzer fills a histogram with the event weights,
   and looks for electron/positron pairs and fills a histogram
   with the invaraint mass of the two. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: ZeeAnalyzer.cc,v 1.10 2012/08/23 21:51:21 wdd Exp $
//
//


// system include files
#include <memory>
#include <iostream>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/WeightContainer.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "TH1D.h"
#include "TFile.h"

//
// class decleration
//

class ZeeAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ZeeAnalyzer(const edm::ParameterSet&);
      ~ZeeAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      
  std::string outputFilename;
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
  edm::InputTag hepMCProductTag_;
  edm::InputTag genEventInfoProductTag_;
};


ZeeAnalyzer::ZeeAnalyzer(const edm::ParameterSet& iConfig) :
  hepMCProductTag_(iConfig.getParameter<edm::InputTag>("hepMCProductTag")),
  genEventInfoProductTag_(iConfig.getParameter<edm::InputTag>("genEventInfoProductTag"))
{

  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");

  sumWeights=0.0;


  weight_histo  = new TH1D("weight_histo","weight_histo",20,-10,10);
  invmass_histo = new TH1D("invmass_histo","invmass_histo",160,70,110);
  Zpt = new TH1D("Zpt","Zpt",200,0,200);
  hardpt = new TH1D("hardpt","hardpt",200,0,200);
  softpt = new TH1D("softpt","softpt",200,0,200);
  hardeta = new TH1D("hardeta","hardeta",200,-5,5);
  softeta = new TH1D("softeta","softeta",200,-5,5);
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
   iEvent.getByLabel(hepMCProductTag_, evt_h);
   HepMC::GenEvent* evt = new  HepMC::GenEvent(*(evt_h->GetEvent()));

   Handle<GenEventInfoProduct> evt_info;
   iEvent.getByLabel(genEventInfoProductTag_, evt_info);


   double weight = evt_info->weight();
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
   double Zpt_ = 0.0;
   if(electrons.size()>1) {
     math::XYZTLorentzVector tot_momentum(electrons[0]->momentum());
     math::XYZTLorentzVector mom2(electrons[1]->momentum());
     tot_momentum += mom2;
     inv_mass = sqrt(tot_momentum.mass2());
     Zpt_=tot_momentum.pt();
     
     // IMPORTANT: use the weight of the event ...
     
     double weight_sign = (weight > 0) ? 1. : -1.;
     invmass_histo->Fill(inv_mass,weight_sign);
     Zpt->Fill(Zpt_,weight_sign);
     if(electrons[0]->momentum().perp()>electrons[1]->momentum().perp()) {
       hardpt->Fill(electrons[0]->momentum().perp(),weight_sign);
       softpt->Fill(electrons[1]->momentum().perp(),weight_sign);
       hardeta->Fill(electrons[0]->momentum().eta(),weight_sign);
       softeta->Fill(electrons[1]->momentum().eta(),weight_sign);
     } else {
       hardpt->Fill(electrons[1]->momentum().perp(),weight_sign);
       softpt->Fill(electrons[0]->momentum().perp(),weight_sign);       
       hardeta->Fill(electrons[1]->momentum().eta(),weight_sign);
       softeta->Fill(electrons[0]->momentum().eta(),weight_sign);
     }

     sumWeights+=weight_sign;
   }

   delete evt;
}


// ------------ method called once each job just before starting event loop  ------------
void 
ZeeAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZeeAnalyzer::endJob() {

  std::cout<<" total sum wieghts = "<<sumWeights<<std::endl;

  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  weight_histo->Write();
  invmass_histo->Write();
  Zpt->Write();
  hardpt->Write();
  softpt->Write();
  hardeta->Write();
  softeta->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeeAnalyzer);
