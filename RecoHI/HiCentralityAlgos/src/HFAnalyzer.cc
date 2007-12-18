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
// $Id: HFAnalyzer.cc,v 1.1 2007/11/19 17:08:16 yilmaz Exp $
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

   int nbins_;
   string src_;
   std::vector<double> vHF;

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
   src_ = iConfig.getUntrackedParameter<string>("signal","hfreco");
   nbins_ = iConfig.getUntrackedParameter<int>("nBins",20);

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
   double eHF = 0;

   Handle<HFRecHitCollection> hits;
   iEvent.getByLabel(src_,hits);
   for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
      const HFRecHit & tower = (*hits)[ ihit ];
      eHF = eHF + tower.energy();
   }

   eHistHF->Fill(eHF);
   vHF.push_back(eHF);
}


// ------------ method called once each job just before starting event loop  ------------
void 
HFAnalyzer::beginJob(const edm::EventSetup&)
{
   eHistHF = fs->make<TH1D>("eHistHF","HF Energy",500,0.,500000.);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HFAnalyzer::endJob() {

   CentralityTable* cent =  new CentralityTable();
   cent->m_table.reserve(nbins_);

   std::sort(vHF.begin(),vHF.end());
   double nev = vHF.size();

   std::cout<<" Number of Events : "<<nev<<std::endl;

   for(int j=0; j<nbins_; j++){
      double bin = vHF[(size_t)nev*0.05*(nbins_-1-j)];
      cout<<"Centrality Bin : "<<j<<";Lower Cut on HF energy : "<<bin<<endl;
      
      CentralityTable::Bins newvalues;
      newvalues.hf_low_cut  = bin;
      newvalues.n_part_mean = 0;
      newvalues.n_part_var  = 0;
      newvalues.n_coll_mean = 0;
      newvalues.n_coll_var  = 0;
      newvalues.b_mean      = 0;
      newvalues.b_var       = 0;
      
      cent->m_table.push_back(newvalues);
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
