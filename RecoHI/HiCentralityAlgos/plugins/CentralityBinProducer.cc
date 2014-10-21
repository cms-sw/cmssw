// -*- C++ -*-
//
// Package:    CentralityBinProducer
// Class:      CentralityBinProducer
// 
/**\class CentralityBinProducer CentralityBinProducer.cc RecoHI/CentralityBinProducer/src/CentralityBinProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Aug 12 05:34:11 EDT 2010
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"


//
// class declaration
//

class CentralityBinProducer : public edm::EDProducer {
   public:
      explicit CentralityBinProducer(const edm::ParameterSet&);
      explicit CentralityBinProducer(const edm::ParameterSet&, const edm::EventSetup&, edm::ConsumesCollector &&);
      ~CentralityBinProducer();

   private:
      virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;
      
      // ----------member data ---------------------------

   CentralityProvider * centrality_;

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
CentralityBinProducer::CentralityBinProducer(const edm::ParameterSet& iConfig){}

CentralityBinProducer::CentralityBinProducer(const edm::ParameterSet& iConfig, const edm::EventSetup& iSetup, edm::ConsumesCollector && iC) :
  centrality_(0)
{
   using namespace edm;
   if(!centrality_) centrality_ = new CentralityProvider(iSetup, std::move(iC));

   produces<int>();  
}


CentralityBinProducer::~CentralityBinProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CentralityBinProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   centrality_->newEvent(iEvent,iSetup);

   int bin = centrality_->getBin();
   std::auto_ptr<int> binp(new int(bin));

   iEvent.put(binp);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
CentralityBinProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CentralityBinProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(CentralityBinProducer);
