// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTestAnalyzer.h"

using std::cout;
using std::endl;

//
// constructors and destructor
//
L1RCTTestAnalyzer::L1RCTTestAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


L1RCTTestAnalyzer::~L1RCTTestAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1RCTTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

   // as in L1GctTestAnalyzer.cc
   Handle<L1CaloEmCollection> rctEmCands;
   Handle<L1CaloRegionCollection> rctRegions;

   L1CaloEmCollection::const_iterator em;
   L1CaloRegionCollection::const_iterator rgn;

   iEvent.getByType(rctEmCands);
   iEvent.getByType(rctRegions);

   cout << endl << endl << "THIS IS NOW THE ANALYZER RUNNING" << endl;
   cout << "EmCand objects" << endl;
   for (em=rctEmCands->begin(); em!=rctEmCands->end(); em++){
     cout << "(Analyzer)\n" << (*em) << endl;
   }
   cout << endl;

   cout << "Regions" << endl;
   for (rgn=rctRegions->begin(); rgn!=rctRegions->end(); rgn++){
     cout << "(Analyzer)\n" << (*rgn) << endl;
   }
   cout << endl;

}
