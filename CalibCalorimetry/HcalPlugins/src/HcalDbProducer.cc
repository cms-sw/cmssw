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
// $Id: HcalDbProducer.cc,v 1.3 2005/10/04 18:03:03 fedor Exp $
//
//


// system include files
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServicePool.h"
//Frontier #include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceFrontier.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"


#include "HcalDbProducer.h"

namespace {
  const std::string name_hardcode ("hardcode");
  const std::string name_frontier ("frontier");
  const std::string name_pool ("pool");
}

HcalDbProducer::HcalDbProducer( const edm::ParameterSet& iConfig )
  :
  mDbSourceName (iConfig.getUntrackedParameter<std::string>("dbSource", name_hardcode)),
  mPedestals (0),
  mPedestalWidths (0),
  mGains (0),
  mGainWidths (0)
{
   //the following line is needed to tell the framework what
   // data is being produced
  std::cout << "HcalDbProducer::HcalDbProducer... Will use dbSource " 
	    << mDbSourceName << std::endl;
   setWhatProduced (this);
   setWhatProduced (this, (dependsOn (&HcalDbProducer::poolPedestalsCallback) &
			   &HcalDbProducer::poolPedestalWidthsCallback &
			   &HcalDbProducer::poolGainsCallback &
			   &HcalDbProducer::poolGainWidthsCallback)
		    ); 

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

  
  const HcalDbServiceBase* service = 0;
  if (mDbSourceName == name_hardcode) {
    service = new HcalDbServiceHardcode ();
  }
  else if (mDbSourceName == name_pool) {
    HcalDbServicePool* poolService = new HcalDbServicePool ();
    service = poolService;
    if (mPedestals) poolService->setPedestals (mPedestals);
    if (mPedestalWidths) poolService->setPedestalWidths (mPedestalWidths);
    if (mGains) poolService->setGains (mGains);
    if (mGainWidths) poolService->setGainWidths (mGainWidths);
  }
  else {
    std::cerr << "HcalDbProducer::produce-> Unknown service " << mDbSourceName
	      << ". Use one of: "  
	      << name_hardcode 
	      << std::endl;
  }
  std::auto_ptr<HcalDbService> pHcalDbService (new HcalDbService (service));

  std::cout << "HcalDbProducer::produce-> got service " << pHcalDbService->service()->name () << std::endl;
  return pHcalDbService;
}

void HcalDbProducer::poolPedestalsCallback (const HcalPedestalsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolPedestalsCallback->..." << std::endl;
  if (mDbSourceName != name_pool) return;
  edm::ESHandle <HcalPedestals> item;
  fRecord.get (item);
  mPedestals = item.product ();
}

  void HcalDbProducer::poolPedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolPedestalWidthsCallback->..." << std::endl;
  if (mDbSourceName != name_pool) return;
  edm::ESHandle <HcalPedestalWidths> item;
  fRecord.get (item);
  mPedestalWidths = item.product ();
}


  void HcalDbProducer::poolGainsCallback (const HcalGainsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolGainsCallback->..." << std::endl;
  if (mDbSourceName != name_pool) return;
  edm::ESHandle <HcalGains> item;
  fRecord.get (item);
  mGains = item.product ();
}


  void HcalDbProducer::poolGainWidthsCallback (const HcalGainWidthsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolGainWidthsCallback->..." << std::endl;
  if (mDbSourceName != name_pool) return;
  edm::ESHandle <HcalGainWidths> item;
  fRecord.get (item);
  mGainWidths = item.product ();
}



