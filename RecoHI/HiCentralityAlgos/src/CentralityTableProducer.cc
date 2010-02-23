// -*- C++ -*-
//
// Package:    CentralityTableProducer
// Class:      CentralityTableProducer
// 
/**\class CentralityTableProducer CentralityTableProducer.cc yetkin/CentralityTableProducer/src/CentralityTableProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Wed May  2 21:41:30 EDT 2007
// $Id: CentralityTableProducer.cc,v 1.2 2007/12/18 13:03:21 yilmaz Exp $
//
//


// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"

#include <TFile.h>

using namespace std;
//
// class decleration
//

class CentralityTableProducer : public edm::EDAnalyzer {
   public:
      explicit CentralityTableProducer(const edm::ParameterSet&);
      ~CentralityTableProducer();

   private:
      virtual void beginRun(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
   void printBin(const CentralityTable::CBin*);
      // ----------member data ---------------------------

   int nbins_;

   bool makeDBFromTFile_;
   bool makeTFileFromDB_;

   edm::ESHandle<CentralityTable> inputDB_;
   TFile* inputTFile_;
   string inputTFileName_;   
   edm::Service<TFileService> fs;

   string rootTag_;
   ofstream text_;

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
CentralityTableProducer::CentralityTableProducer(const edm::ParameterSet& iConfig):
   text_("results.txt")
{
   //now do what ever initialization is needed
   makeDBFromTFile_ = iConfig.getUntrackedParameter<bool>("makeDBFromTFile",1);
   makeTFileFromDB_ = iConfig.getUntrackedParameter<bool>("makeTFileFromDB",0);
   if(makeDBFromTFile_){
      inputTFileName_ = iConfig.getParameter<string>("inputTFile");
      rootTag_ = iConfig.getParameter<string>("rootTag");
   }
   nbins_ = iConfig.getParameter<int>("nBins");
   if(makeDBFromTFile_){
      inputTFile_  = new TFile(inputTFileName_.data(),"read");
      cout<<inputTFileName_.data()<<endl;
   }

}


CentralityTableProducer::~CentralityTableProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CentralityTableProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // do nothing

   if(makeTFileFromDB_ && !inputDB_.isValid()){
      iSetup.get<HeavyIonRcd>().get(inputDB_);
      nbins_ =  (*inputDB_).m_table.size();
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
CentralityTableProducer::beginRun(const edm::EventSetup& iSetup)
{
   cout<<"Beginning Run"<<endl;
   // Get Heavy Ion Record
   if(makeTFileFromDB_){
      iSetup.get<HeavyIonRcd>().get(inputDB_);
      nbins_ =  (*inputDB_).m_table.size();
   }
   cout<<"Centrality Values are being determined for "<<nbins_<<" HF energy bins of equal cross section."<<endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CentralityTableProducer::endJob() {

   CentralityTable* CT;
   CentralityBins* CB;

   if(makeDBFromTFile_){
      // Get values from root file
      CB = (CentralityBins*) inputTFile_->Get(rootTag_.data());
      cout<<rootTag_.data()<<endl;
      CT = new CentralityTable();
      CT->m_table.reserve(nbins_);
   }

   if(makeTFileFromDB_){
      CB = fs->make<CentralityBins>(rootTag_.data(),"",nbins_);
   }

   for(int j=0; j<nbins_; j++){
    if(makeDBFromTFile_){
       CentralityTable::CBin* thisBin = new CentralityTable::CBin();
       thisBin->bin_edge = CB->lowEdgeOfBin(j);
       thisBin->n_part_mean = CB->NpartMeanOfBin(j);
       thisBin->n_part_var  = CB->NpartSigmaOfBin(j);
       thisBin->n_coll_mean = CB->NcollMeanOfBin(j);
       thisBin->n_coll_var  = CB->NcollSigmaOfBin(j);
       thisBin->n_hard_mean = CB->NhardMeanOfBin(j);
       thisBin->n_hard_var  = CB->NhardSigmaOfBin(j);
       thisBin->b_mean = CB->bMeanOfBin(j);
       thisBin->b_var = CB->bSigmaOfBin(j);
       printBin(thisBin);
       CT->m_table.push_back(*thisBin);
       if(thisBin) delete thisBin;
    }
   
    if(makeTFileFromDB_){
       const CentralityTable::CBin* thisBin;
       thisBin = &(inputDB_->m_table[j]); 
       CB->table_[j].bin_edge = thisBin->bin_edge;
       CB->table_[j].n_part_mean = thisBin->n_part_mean;
       CB->table_[j].n_part_var  = thisBin->n_part_var;
       CB->table_[j].n_coll_mean = thisBin->n_coll_mean;
       CB->table_[j].n_coll_var  = thisBin->n_coll_var;
       CB->table_[j].n_hard_mean = thisBin->n_hard_mean;
       CB->table_[j].n_hard_var  = thisBin->n_hard_var;
       CB->table_[j].b_mean = thisBin->b_mean;
       CB->table_[j].b_var = thisBin->b_var;
       printBin(thisBin);
    }
   }

   if(makeDBFromTFile_){
      edm::Service<cond::service::PoolDBOutputService> pool;
      if( pool.isAvailable() )
	 if( pool->isNewTagRequest( "HeavyIonRcd" ) )
	    pool->createNewIOV<CentralityTable>( CT, pool->beginOfTime(), pool->endOfTime(), "HeavyIonRcd" );
	 else
	    pool->appendSinceTime<CentralityTable>( CT, pool->currentTime(), "HeavyIonRcd" );
   }

}


void CentralityTableProducer::printBin(const CentralityTable::CBin* thisBin){
   
   cout<<"HF Cut = "<<thisBin->bin_edge<<endl;
   cout<<"Npart = "<<thisBin->n_part_mean<<endl;
   cout<<"sigma = "<<thisBin->n_part_var<<endl;
   cout<<"Ncoll = "<<thisBin->n_coll_mean<<endl;
   cout<<"sigma = "<<thisBin->n_coll_var<<endl;
   cout<<"B     = "<<thisBin->b_mean<<endl;
   cout<<"sigma = "<<thisBin->b_var<<endl;
   text_<<Form("%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.1f",
	       (Float_t)thisBin->n_part_mean,
	       (Float_t)thisBin->n_part_var,
	       (Float_t)thisBin->n_coll_mean,
	       (Float_t)thisBin->n_coll_var,
	       (Float_t)thisBin->b_mean,
	       (Float_t)thisBin->b_var,
	       (Float_t)thisBin->bin_edge)
	<<endl;
   cout<<"__________________________________________________"<<endl;
   
}



//define this as a plug-in
DEFINE_FWK_MODULE(CentralityTableProducer);
