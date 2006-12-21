// -*- C++ -*-
//
// Package:    L1GeometryProducers
// Class:      L1CaloGeometryProd
// 
/**\class L1CaloGeometryProd L1CaloGeometryProd.h L1TriggerConfig/L1GeometryProducers/interface/L1CaloGeometryProd.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Sun
//         Created:  Tue Oct 24 00:00:00 EDT 2006
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "L1TriggerConfig/L1GeometryProducers/interface/L1CaloGeometryProd.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1CaloGeometryProd::L1CaloGeometryProd(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


L1CaloGeometryProd::~L1CaloGeometryProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1CaloGeometryProd::ReturnType
L1CaloGeometryProd::produce(const L1CaloGeometryRecord& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<L1CaloGeometry> pL1CaloGeometry ;

   pL1CaloGeometry = std::auto_ptr< L1CaloGeometry >(
      new L1CaloGeometry() ) ;

   return pL1CaloGeometry ;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1CaloGeometryProd)
