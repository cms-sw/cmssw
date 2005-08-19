// -*- C++ -*-
//
// Package:    HcalDbProducer
// Class:      HcalDbProducer
// 
/**\class HcalDbProducer HcalDbProducer.h CalibFormats/HcalDbProducer/interface/HcalDbProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Tue Aug  9 19:10:10 CDT 2005
// $Id: HcalDbProducer.cc,v 1.1 2005/08/18 23:45:05 fedor Exp $
//
//


// system include files
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
//Frontier #include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceFrontier.h"

#include "HcalDbProducer.h"

namespace {
  const std::string name_hardcode ("hardcode");
  const std::string name_frontier ("frontier");
  const std::string name_pool ("pool");
}

HcalDbProducer::HcalDbProducer( const edm::ParameterSet& iConfig )
  :
  mDbSourceName (iConfig.getUntrackedParameter<std::string>("dbSource", name_hardcode))
{
   //the following line is needed to tell the framework what
   // data is being produced
  std::cout << "HcalDbProducer::HcalDbProducer... Will use dbSource " 
	    << mDbSourceName << std::endl;
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


HcalDbProducer::~HcalDbProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
HcalDbProducer::ReturnType
HcalDbProducer::produce( const HcalDbRecord& fRecord )
{
  std::cout << "HcalDbProducer::produce..." << std::endl;

  
  const HcalDbService* service = 0;
  if (mDbSourceName == name_hardcode) {
    edm::eventsetup::ESHandle <HcalDbServiceHardcode> serviceHardcode;
    fRecord.get (serviceHardcode);
    service = serviceHardcode.product();
  }
//Frontier  else if (mDbSourceName == name_frontier) {
//Frontier    edm::eventsetup::ESHandle <HcalDbServiceFrontier> serviceFrontier;
//Frontier    fRecord.get (serviceFrontier);
//Frontier    service = serviceFrontier.product();
//Frontier  }
  else {
    std::cerr << "HcalDbProducer::produce-> Unknown service " << mDbSourceName
	      << ". Use one of: "  
	      << name_hardcode 
//Frontier	      << ", " << name_frontier  
	      << std::endl;
  }
  std::auto_ptr<HcalDbServiceHandle> pHcalDbServiceHandle (new HcalDbServiceHandle (service));

  std::cout << "HcalDbProducer::produce-> got service " << pHcalDbServiceHandle->service()->name () << std::endl;
  return pHcalDbServiceHandle ;
}

