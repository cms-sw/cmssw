#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationProducers.h"


EgammaHLTHcalIsolationProducers::EgammaHLTHcalIsolationProducers(const edm::ParameterSet& iConfig)
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
   //now do what ever other initialization is needed

}


EgammaHLTHcalIsolationProducers::~EgammaHLTHcalIsolationProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EgammaHLTHcalIsolationProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
}

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTHcalIsolationProducers)
