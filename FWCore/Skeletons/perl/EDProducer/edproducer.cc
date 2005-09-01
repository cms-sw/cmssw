// -*- C++ -*-
//
// Package:    prodname
// Class:      prodname
// 
/**\class prodname prodname.cc skelsubsys/prodname/src/prodname.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
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

//
// class decleration
//

class prodname : public edm::EDProducer {
   public:
      explicit prodname(const edm::ParameterSet&);
      ~prodname();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
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
prodname::prodname(const edm::ParameterSet& iConfig)
{
   //register your products
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
#endif

   //now do what ever other initialization is needed

}


prodname::~prodname()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
prodname::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   auto_ptr<ExampleData2> pOut(new ExampleData2());
   iEvent.put(pOut);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}

//define this as a plug-in
DEFINE_FWK_MODULE(prodname)
