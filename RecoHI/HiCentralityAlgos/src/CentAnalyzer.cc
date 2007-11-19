// -*- C++ -*-
//
// Package:    CentAnalyzer
// Class:      CentAnalyzer
// 
/**\class CentAnalyzer CentAnalyzer.cc yetkin/CentAnalyzer/src/CentAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Wed May  2 21:41:30 EDT 2007
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"


#include <TNtuple.h>
#include <TH1D.h>
#include <TFile.h>

using namespace std;
//
// class decleration
//

class CentAnalyzer : public edm::EDAnalyzer {
   public:
      explicit CentAnalyzer(const edm::ParameterSet&);
      ~CentAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

   double eta;
   double eFwd;
   double eHF;
   double bin[20];
   int nColl;
   int nPart;
   float b;

   TNtuple *nTup;
   TH1D *eHistE;
   TH1D *eHistHF;
   TH1D *newHist;

vector<TH1D*> NpartHist;
vector<TH1D*> NcollHist;
vector<TH1D*> BHist;

edm::ESHandle<CentralityTable> hfinput;
edm::Service<TFileService> fs;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CentAnalyzer::CentAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


CentAnalyzer::~CentAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace HepMC;

   //Reset energy calculation
   eFwd = 0;
   eHF = 0;
  
   Handle<CandidateCollection> genParticles;
   iEvent.getByLabel("genParticleCandidates", genParticles);
   for( size_t ipar = 0; ipar < genParticles->size(); ++ ipar ) {
      const Candidate & p = (*genParticles)[ ipar ];
      eta = p.eta();
 
      // APPLY SELECTION HERE /////////////
      
      if((p.numberOfDaughters() == 0) && (eta*eta <25.) && (eta*eta >9.)){ 
         eFwd = eFwd + p.energy();
      }
   }     

   Handle<HFRecHitCollection> hits;
   iEvent.getByLabel("hfreco",hits);
   for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
      const HFRecHit & rechit = (*hits)[ ihit ];
      eHF = eHF + rechit.energy();
   }
   

   /////////////////////////////////

   Handle<HepMCProduct> hepmc;
   iEvent.getByLabel("source",hepmc);
   const GenEvent *ev = hepmc->GetEvent();
   HeavyIon *hi;
   hi = ev->heavy_ion();

   nColl = hi->Ncoll();
   nPart = hi->Npart_proj() + hi->Npart_targ();
   b = hi->impact_parameter();

   eHistE->Fill(eFwd);
  
   // FILL HISTOGRAMS /////////////////
   
   eHistHF->Fill(eHF);
   nTup->Fill(eFwd,eHF,nColl,nPart,b);

   eHistHF->Fill(eHF);
   nTup->Fill(eFwd,eHF,nColl,nPart,b);

   cout<<"values extracted"<<endl;
   cout<<"size = "<<(*hfinput).m_table.size()<<endl;
   for(int ie = 1; ie<20; ++ie){
      double Emax = (*hfinput).m_table[ie-1].hf_low_cut;
      double Emin = (*hfinput).m_table[ie].hf_low_cut;
      if(eHF>Emin && eHF<Emax){
         NpartHist[ie]->Fill(nPart);
         NcollHist[ie]->Fill(nColl);
         BHist[ie]->Fill(b);
      }
   }

   if(eHF>(*hfinput).m_table[0].hf_low_cut){
     NpartHist[0]->Fill(nPart);
     NcollHist[0]->Fill(nColl);
     BHist[0]->Fill(b);
     }

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
CentAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
   nTup = fs->make<TNtuple>("nTup","Super Ntuple eFwd","eFwd:eHF:nColl:nPart:b");
   eHistE = fs->make<TH1D>("eHistE","Forward Particles Energy",800,0.,500000.);
   eHistHF = fs->make<TH1D>("eHistHF","HF Energy",100000,0.,500000.);

for(int i=0; i<20; ++i){
TString npname(Form("NpartHist%d",i));
TString ncname(Form("NcollHist%d",i));
TString bname(Form("BHist%d",i));
TH1D *NpHist = fs->make<TH1D>(npname.Data(),";Npart;#",200,0.,450.);
TH1D *NcHist = fs->make<TH1D>(ncname.Data(),";Ncoll;#",200,0.,2000.);
TH1D *Bist = fs->make<TH1D>(bname.Data(),";B;#",200,0.,13.5);

NpartHist.push_back(NpHist);
NcollHist.push_back(NcHist);
BHist.push_back(Bist);
}

iSetup.get<HeavyIonRcd>().get(hfinput);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CentAnalyzer::endJob() {

ofstream text("NcutTest.txt");

CentralityTable CT;
CT.m_table.reserve(20);

   for(int j=0; j<20; j++){
      CentralityTable::Bins newCuts; 
            newCuts.hf_low_cut  = (*hfinput).m_table[j].hf_low_cut;
            newCuts.n_part_mean = NpartHist[j]->GetMean();
            newCuts.n_part_var  = NpartHist[j]->GetRMS();
            newCuts.n_coll_mean = NcollHist[j]->GetMean();
            newCuts.n_coll_var  = NcollHist[j]->GetRMS();
            newCuts.b_mean = BHist[j]->GetMean();
            newCuts.b_var = BHist[j]->GetRMS();


cout<<"Bin = "<<j<<endl;
cout<<"HF Cut = "<<newCuts.hf_low_cut<<endl;
cout<<"Npart = "<<newCuts.n_part_mean<<endl;
cout<<"sigma = "<<newCuts.n_part_var<<endl;
cout<<"Ncoll = "<<newCuts.n_coll_mean<<endl;
cout<<"sigma = "<<newCuts.n_coll_var<<endl;
cout<<"B     = "<<newCuts.b_mean<<endl;
cout<<"sigma = "<<newCuts.b_var<<endl;
text<<Form("%d-%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.1f",100-5*j,95-5*j,(Float_t)newCuts.n_part_mean,(Float_t)newCuts.n_part_var,(Float_t)newCuts.n_coll_mean,(Float_t)newCuts.n_coll_var,(Float_t)newCuts.b_mean,(Float_t)newCuts.b_var,(Float_t)newCuts.hf_low_cut)<<endl;
cout<<"__________________________________________________"<<endl;

CT.m_table.push_back(newCuts);
   }
CentralityTable *cent = new CentralityTable(CT);
   // OUTPUT DATA
                                                                                                                                                                                                       
   edm::Service<cond::service::PoolDBOutputService> pool;
   if( pool.isAvailable() )
     if( pool->isNewTagRequest( "HeavyIonRcd" ) )
       pool->createNewIOV<CentralityTable>( cent, pool->endOfTime(), "HeavyIonRcd" );
     else
       pool->appendSinceTime<CentralityTable>( cent, pool->currentTime(), "HeavyIonRcd" );

}

//define this as a plug-in
DEFINE_FWK_MODULE(CentAnalyzer);
