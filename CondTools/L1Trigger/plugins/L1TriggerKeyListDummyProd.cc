// -*- C++ -*-
//
// Package:    L1TriggerKeyListDummyProd
// Class:      L1TriggerKeyListDummyProd
// 
/**\class L1TriggerKeyListDummyProd L1TriggerKeyListDummyProd.h CondTools/L1TriggerKeyListDummyProd/src/L1TriggerKeyListDummyProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Sat Mar  1 05:02:13 CET 2008
// $Id: L1TriggerKeyListDummyProd.cc,v 1.1 2008/03/03 21:52:18 wsun Exp $
//
//


// system include files

// user include files
#include "CondTools/L1Trigger/plugins/L1TriggerKeyListDummyProd.h"

//
// class declaration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TriggerKeyListDummyProd::L1TriggerKeyListDummyProd(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


L1TriggerKeyListDummyProd::~L1TriggerKeyListDummyProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyListDummyProd::ReturnType
L1TriggerKeyListDummyProd::produce(const L1TriggerKeyListRcd& iRecord)
{
   using namespace edm::es;
   boost::shared_ptr<L1TriggerKeyList> pL1TriggerKeyList ;

   pL1TriggerKeyList = boost::shared_ptr< L1TriggerKeyList >(
      new L1TriggerKeyList() ) ;

   return pL1TriggerKeyList ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyListDummyProd);
