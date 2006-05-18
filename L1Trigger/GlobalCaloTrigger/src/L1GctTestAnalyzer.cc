
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigi.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctTestAnalyzer.h"

//
// constructors and destructor
//
L1GctTestAnalyzer::L1GctTestAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


L1GctTestAnalyzer::~L1GctTestAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1GctTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<L1GctIsoEmCollection> pIn;
   iEvent.getByLabel("example",pIn);

   // get some GCT digis
   

}

