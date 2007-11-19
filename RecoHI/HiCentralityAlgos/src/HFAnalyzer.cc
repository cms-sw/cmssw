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
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"


#include <TNtuple.h>
#include <TH1.h>
#include <TFile.h>

using namespace std;
//
// class decleration
//

class HFAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HFAnalyzer(const edm::ParameterSet&);
      ~HFAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

   double eta;
   double eFwd;
   double eHF;
   int nColl;
   int nPart;
   float b;

   TH1 *eHistHF;
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
HFAnalyzer::HFAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


HFAnalyzer::~HFAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HFAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace HepMC;

   //Reset energy calculation
   eHF = 0;

   Handle<HFRecHitCollection> hits;
   iEvent.getByLabel("hfreco",hits);
   for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
      const HFRecHit & tower = (*hits)[ ihit ];
      eHF = eHF + tower.energy();
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

   eHistHF->Fill(eHF);
  
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
HFAnalyzer::beginJob(const edm::EventSetup&)
{
   eHistHF = fs->make<TH1D>("eHistHF","HF Energy",100000,0.,500000.);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HFAnalyzer::endJob() {

CentralityTable* cent =  new CentralityTable();
cent->m_table.reserve(20);

   double I = eHistHF->GetEntries();
   for(int j=0; j<20; j++){
      double f = 0;
      for(int in = eHistHF->GetNbinsX(); in>0; in--){
         f = f + eHistHF->GetBinContent(in);
         if((f/I)>=0.05*(j+1)){
            double bin = eHistHF->GetBinLowEdge(in);
            cout<<"Centrality Bin:"<<in<<";Lower Cut:"<<bin<<endl;
       
            CentralityTable::Bins newvalues;
            newvalues.hf_low_cut  = bin;
            newvalues.n_part_mean = 0;
            newvalues.n_part_var  = 0;
            newvalues.n_coll_mean = 0;
            newvalues.n_coll_var  = 0;
            newvalues.b_mean      = 0;
            newvalues.b_var       = 0;

            cent->m_table.push_back(newvalues);

            break;
         }
      }
   }

   // OUTPUT DATA
   edm::Service<cond::service::PoolDBOutputService> pool;
   if( pool.isAvailable() )
     if( pool->isNewTagRequest( "HeavyIonRcd" ) )
       pool->createNewIOV<CentralityTable>( cent, pool->endOfTime(), "HeavyIonRcd" );
     else
       pool->appendSinceTime<CentralityTable>( cent, pool->currentTime(), "HeavyIonRcd" );

}

//define this as a plug-in
DEFINE_FWK_MODULE(HFAnalyzer);
