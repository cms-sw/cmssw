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
// $Id: HcalDbProducer.cc,v 1.19 2008/03/03 20:22:36 rofierzy Exp $
//
//


// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"


#include "CondFormats/HcalObjects/interface/AllObjects.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "HcalDbProducer.h"

HcalDbProducer::HcalDbProducer( const edm::ParameterSet& fConfig)
  : ESProducer(),
    mService (new HcalDbService (fConfig)),
    mDumpRequest (),
    mDumpStream(0)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced (this, (dependsOn (&HcalDbProducer::pedestalsCallback) &
			  &HcalDbProducer::pedestalWidthsCallback &
			  &HcalDbProducer::gainsCallback &
			  &HcalDbProducer::gainWidthsCallback &
			  &HcalDbProducer::QIEDataCallback &
			  &HcalDbProducer::channelQualityCallback &
			  &HcalDbProducer::zsThresholdsCallback &
			  &HcalDbProducer::respCorrsCallback &
			  &HcalDbProducer::electronicsMapCallback
			  )
		   );
  
  //now do what ever other initialization is needed

  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  if (!mDumpRequest.empty()) {
    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  }
}


HcalDbProducer::~HcalDbProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (mDumpStream != &std::cout) delete mDumpStream;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
boost::shared_ptr<HcalDbService> HcalDbProducer::produce( const HcalDbRecord&)
{
  return mService;
}

void HcalDbProducer::pedestalsCallback (const HcalPedestalsRcd& fRecord) {
  edm::ESHandle <HcalPedestals> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::pedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord) {
  edm::ESHandle <HcalPedestalWidths> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


void HcalDbProducer::gainsCallback (const HcalGainsRcd& fRecord) {
  edm::ESHandle <HcalGains> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


void HcalDbProducer::gainWidthsCallback (const HcalGainWidthsRcd& fRecord) {
  edm::ESHandle <HcalGainWidths> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::QIEDataCallback (const HcalQIEDataRcd& fRecord) {
  edm::ESHandle <HcalQIEData> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::channelQualityCallback (const HcalChannelQualityRcd& fRecord) {
  edm::ESHandle <HcalChannelQuality> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::respCorrsCallback (const HcalRespCorrsRcd& fRecord) {
  edm::ESHandle <HcalRespCorrs> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RespCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::zsThresholdsCallback (const HcalZSThresholdsRcd& fRecord) {
  edm::ESHandle <HcalZSThresholds> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZSThresholds")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::electronicsMapCallback (const HcalElectronicsMapRcd& fRecord) {
  edm::ESHandle <HcalElectronicsMap> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Electronics Map set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}



