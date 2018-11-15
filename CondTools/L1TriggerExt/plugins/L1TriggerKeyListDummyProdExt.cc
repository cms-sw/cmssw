#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyListDummyProdExt.h"

L1TriggerKeyListDummyProdExt::L1TriggerKeyListDummyProdExt(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


L1TriggerKeyListDummyProdExt::~L1TriggerKeyListDummyProdExt()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyListDummyProdExt::ReturnType
L1TriggerKeyListDummyProdExt::produce(const L1TriggerKeyListExtRcd& iRecord)
{
   using namespace edm::es;
   return std::make_unique< L1TriggerKeyListExt >() ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyListDummyProdExt);
