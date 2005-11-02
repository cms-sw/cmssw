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
// $Id: HcalDbProducer.cc,v 1.5 2005/10/28 01:30:47 fedor Exp $
//
//


// system include files
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"

#include "CondFormats/DataRecord/interface/AllHcalRecords.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"


#include "HcalDbProducer.h"

HcalDbProducer::HcalDbProducer( const edm::ParameterSet&)
  : mService (new HcalDbService ())
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced (this);
  setWhatProduced (this, (dependsOn (&HcalDbProducer::pedestalsCallback) &
			  &HcalDbProducer::pedestalWidthsCallback &
			  &HcalDbProducer::gainsCallback &
			  &HcalDbProducer::gainWidthsCallback &
			  &HcalDbProducer::QIEShapeCallback &
			  &HcalDbProducer::QIEDataCallback &
			  &HcalDbProducer::channelQualityCallback &
			  &HcalDbProducer::electronicsMapCallback
			  )
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
boost::shared_ptr<HcalDbService> HcalDbProducer::produce( const HcalDbRecord&)
{
  std::cout << "HcalDbProducer::produce..." << std::endl;
  return mService;
}

void HcalDbProducer::pedestalsCallback (const HcalPedestalsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolPedestalsCallback->..." << std::endl;
  edm::ESHandle <HcalPedestals> item;
  fRecord.get (item);
  mService->setData (item.product ());
}

  void HcalDbProducer::pedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolPedestalWidthsCallback->..." << std::endl;
  edm::ESHandle <HcalPedestalWidths> item;
  fRecord.get (item);
  mService->setData (item.product ());
}


  void HcalDbProducer::gainsCallback (const HcalGainsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolGainsCallback->..." << std::endl;
  edm::ESHandle <HcalGains> item;
  fRecord.get (item);
  mService->setData (item.product ());
}


  void HcalDbProducer::gainWidthsCallback (const HcalGainWidthsRcd& fRecord) {
  std::cout << "HcalDbProducer::poolGainWidthsCallback->..." << std::endl;
  edm::ESHandle <HcalGainWidths> item;
  fRecord.get (item);
  mService->setData (item.product ());
}

void HcalDbProducer::QIEShapeCallback (const HcalQIEShapeRcd& fRecord) {
  std::cout << "HcalDbProducer::QIEShapeCallback->..." << std::endl;
  edm::ESHandle <HcalQIEShape> item;
  fRecord.get (item);
  mService->setData (item.product ());
}

void HcalDbProducer::QIEDataCallback (const HcalQIEDataRcd& fRecord) {
  std::cout << "HcalDbProducer::QIEDataCallback->..." << std::endl;
  edm::ESHandle <HcalQIEData> item;
  fRecord.get (item);
  mService->setData (item.product ());
}

void HcalDbProducer::channelQualityCallback (const HcalChannelQualityRcd& fRecord) {
  std::cout << "HcalDbProducer::channelQualityCallback->..." << std::endl;
  edm::ESHandle <HcalChannelQuality> item;
  fRecord.get (item);
  mService->setData (item.product ());
}

void HcalDbProducer::electronicsMapCallback (const HcalElectronicsMapRcd& fRecord) {
  std::cout << "HcalDbProducer::electronicsMapCallback->..." << std::endl;
  edm::ESHandle <HcalElectronicsMap> item;
  fRecord.get (item);
  mService->setData (item.product ());
}



