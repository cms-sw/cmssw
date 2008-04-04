// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidths.h"
#include "CondFormats/CastorObjects/interface/CastorGains.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"

#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h" 
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"  	 
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h" 
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"

#include "CastorDbProducer.h"

CastorDbProducer::CastorDbProducer( const edm::ParameterSet& fConfig)
  : ESProducer(),
    mService (new CastorDbService ()),
    mDumpRequest (),
    mDumpStream(0)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced (this, (dependsOn (&CastorDbProducer::pedestalsCallback) &
			  &CastorDbProducer::pedestalWidthsCallback &
			  &CastorDbProducer::gainsCallback &
			  &CastorDbProducer::gainWidthsCallback &
			  &CastorDbProducer::QIEDataCallback &
			  &CastorDbProducer::channelQualityCallback &
			  &CastorDbProducer::electronicsMapCallback
			  )
		   );
  
  //now do what ever other initialization is needed

  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  if (!mDumpRequest.empty()) {
    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  }
}


CastorDbProducer::~CastorDbProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (mDumpStream != &std::cout) delete mDumpStream;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
boost::shared_ptr<CastorDbService> CastorDbProducer::produce( const CastorDbRecord&)
{
  return mService;
}

void CastorDbProducer::pedestalsCallback (const CastorPedestalsRcd& fRecord) {
  edm::ESHandle <CastorPedestals> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

  void CastorDbProducer::pedestalWidthsCallback (const CastorPedestalWidthsRcd& fRecord) {
  edm::ESHandle <CastorPedestalWidths> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


  void CastorDbProducer::gainsCallback (const CastorGainsRcd& fRecord) {
  edm::ESHandle <CastorGains> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


  void CastorDbProducer::gainWidthsCallback (const CastorGainWidthsRcd& fRecord) {
  edm::ESHandle <CastorGainWidths> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::QIEDataCallback (const CastorQIEDataRcd& fRecord) {
  edm::ESHandle <CastorQIEData> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::channelQualityCallback (const CastorChannelQualityRcd& fRecord) {
  edm::ESHandle <CastorChannelQuality> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::electronicsMapCallback (const CastorElectronicsMapRcd& fRecord) {
  edm::ESHandle <CastorElectronicsMap> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Electronics Map set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}



