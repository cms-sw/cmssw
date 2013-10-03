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
// $Id: CentralityBinProducer.cc,v 1.4 2010/11/02 16:15:59 yilmaz Exp $
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
#include "DataFormats/HeavyIonEvent/interface/CentralityProvider.h"


//
// class declaration
//

class CentralityBinProducer : public edm::EDProducer {
   public:
      explicit CentralityBinProducer(const edm::ParameterSet&);
      ~CentralityBinProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
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
CentralityBinProducer::CentralityBinProducer(const edm::ParameterSet& iConfig) :
  centrality_(0)
{
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
   using namespace edm;
   if(!centrality_) centrality_ = new CentralityProvider(iSetup);
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
