// -*- C++ -*-
//
// Package:    GctFakeJetProducer
// Class:      GctFakeJetProducer
// 
/**\class GctFakeJetProducer GctFakeJetProducer.cc EventFilter/GctFakeJetProducer/src/GctFakeJetProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Tue May 27 15:35:44 CEST 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

//
// class decleration
//

class GctFakeJetProducer : public edm::EDProducer {
   public:
      explicit GctFakeJetProducer(const edm::ParameterSet&);
      ~GctFakeJetProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
  edm::InputTag inputTag_;
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
GctFakeJetProducer::GctFakeJetProducer(const edm::ParameterSet& iConfig) :
  inputTag_(iConfig.getParameter<edm::InputTag>("inputTag"))
{
  //register your products
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctEtTotal>();
  produces<L1GctEtHad>();
  produces<L1GctEtMiss>();
  produces<L1GctJetCounts>();

}


GctFakeJetProducer::~GctFakeJetProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GctFakeJetProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
   using namespace edm;

   // get GCT digis
   edm::Handle<L1GctEmCandCollection> isoEm;
   e.getByLabel(inputTag_.label(), "isoEm", isoEm);
   edm::Handle<L1GctEmCandCollection> nonIsoEm;
   e.getByLabel(inputTag_.label(), "nonIsoEm", nonIsoEm);
   edm::Handle<L1GctJetCandCollection> cenJets;
   e.getByLabel(inputTag_.label(), "cenJets", cenJets);
   edm::Handle<L1GctJetCandCollection> forJets;
   e.getByLabel(inputTag_.label(), "forJets", forJets);
   edm::Handle<L1GctJetCandCollection> tauJets;
   e.getByLabel(inputTag_.label(), "tauJets", tauJets);
   edm::Handle<L1GctJetCounts> jetCounts;
   e.getByLabel(inputTag_.label(), "", jetCounts);
   edm::Handle<L1GctEtTotal> etTotal;
   e.getByLabel(inputTag_.label(), "", etTotal);
   edm::Handle<L1GctEtHad> etHad;
   e.getByLabel(inputTag_.label(), "", etHad);
   edm::Handle<L1GctEtMiss> etMiss;
   e.getByLabel(inputTag_.label(), "", etMiss);


   // copy EM
   std::auto_ptr<L1GctEmCandCollection>  gctIsoEm   ( new L1GctEmCandCollection() );
   gctIsoEm->reserve(4);
   std::auto_ptr<L1GctEmCandCollection>  gctNonIsoEm   ( new L1GctEmCandCollection() );
   gctNonIsoEm->reserve(4);
   for (unsigned i=0; i<4; i++) {
     gctIsoEm->push_back(isoEm->at(i));
     gctNonIsoEm->push_back(nonIsoEm->at(i));
   }

   // create jets from electrons
   std::auto_ptr<L1GctJetCandCollection> gctCenJets ( new L1GctJetCandCollection() );
   gctCenJets->reserve(4);
   std::auto_ptr<L1GctJetCandCollection> gctForJets ( new L1GctJetCandCollection() );
   gctForJets->reserve(4);
   std::auto_ptr<L1GctJetCandCollection> gctTauJets ( new L1GctJetCandCollection() );
   gctTauJets->reserve(4);

   L1GctEmCand em;
   for (unsigned i=0; i<4; i++) {
     em = nonIsoEm->at(i);
     gctCenJets->push_back(L1GctJetCand(em.rank(), em.phiIndex(), em.etaIndex(), false, false, 0, 0, 0));
     em = isoEm->at(i);
     gctTauJets->push_back(L1GctJetCand(em.rank(), em.phiIndex(), em.etaIndex(), true, false, 0, 0, 0));
     gctForJets->push_back(L1GctJetCand(0, 0, 0, false, true, 0, 0, 0));
   }

   // copy jet counts and energy sums
   std::auto_ptr<L1GctJetCounts> gctJetCounts( new L1GctJetCounts() );
   std::auto_ptr<L1GctEtTotal> gctEtTotal( new L1GctEtTotal() );
   std::auto_ptr<L1GctEtHad> gctEtHad( new L1GctEtHad() );
   std::auto_ptr<L1GctEtMiss> gctEtMiss( new L1GctEtMiss() );
   

   // copy collections
   e.put(gctIsoEm, "isoEm");
   e.put(gctNonIsoEm, "nonIsoEm");
   e.put(gctCenJets,"cenJets");
   e.put(gctForJets,"forJets");
   e.put(gctTauJets,"tauJets");
   e.put(gctJetCounts);
   e.put(gctEtTotal);
   e.put(gctEtHad);
   e.put(gctEtMiss);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
GctFakeJetProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GctFakeJetProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GctFakeJetProducer);
