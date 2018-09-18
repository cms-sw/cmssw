// -*- C++ -*-
//
// Package:    CastorDbProducer
// Class:      CastorDbProducer
// 
/**\class CastorDbProducer CastorDbProducer.h CalibFormats/CastorDbProducer/interface/CastorDbProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Tue Aug  9 19:10:10 CDT 2005
//         Adapted for CASTOR by L. Mundim
//
//


// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"


#include "CondFormats/CastorObjects/interface/AllObjects.h"

#include "CastorDbProducer.h"

CastorDbProducer::CastorDbProducer( const edm::ParameterSet& fConfig)
  : ESProducer(),
    mDumpRequest (),
    mDumpStream(nullptr)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);

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
std::shared_ptr<CastorDbService> CastorDbProducer::produce( const CastorDbRecord& record)
{
  auto host = holder_.makeOrGet([]() {
    return new HostType;
  });

  bool needBuildCalibrations = false;
  bool needBuildCalibWidths = false;

  host->ifRecordChanges<CastorElectronicsMapRcd>(record,
                                                 [this, h=host.get()](auto const& rec) {
    setupElectronicsMap(rec, h);
  });
  host->ifRecordChanges<CastorChannelQualityRcd>(record,
                                                 [this, h=host.get()](auto const& rec) {
    setupChannelQuality(rec, h);
  });
  host->ifRecordChanges<CastorGainWidthsRcd>(record,
                                             [this, h=host.get(),
                                              &needBuildCalibWidths](auto const& rec) {
    setupGainWidths(rec, h);
    needBuildCalibWidths = true;
  });
  host->ifRecordChanges<CastorQIEDataRcd>(record,
                                          [this, h=host.get(),
                                           &needBuildCalibrations,
                                           &needBuildCalibWidths](auto const& rec) {
    setupQIEData(rec, h);
    needBuildCalibrations = true;
    needBuildCalibWidths = true;
  });
  host->ifRecordChanges<CastorPedestalWidthsRcd>(record,
                                                 [this, h=host.get(),
                                                  &needBuildCalibWidths](auto const& rec) {
    setupPedestalWidths(rec, h);
    needBuildCalibWidths = true;
  });
  host->ifRecordChanges<CastorGainsRcd>(record,
                                        [this, h=host.get(),
                                         &needBuildCalibrations](auto const& rec) {
    setupGains(rec, h);
    needBuildCalibrations = true;
  });
  host->ifRecordChanges<CastorPedestalsRcd>(record,
                                            [this, h=host.get(),
                                             &needBuildCalibrations](auto const& rec) {
    setupPedestals(rec, h);
    needBuildCalibrations = true;
  });

  if (needBuildCalibWidths) {
    host->buildCalibWidths();
  }

  if (needBuildCalibrations) {
    host->buildCalibrations();
  }

  return host; // automatically converts to std::shared_ptr<CastorDbService>
}

void CastorDbProducer::setupPedestals(const CastorPedestalsRcd& fRecord,
                                      CastorDbService* service) {

  edm::ESHandle <CastorPedestals> item;
  fRecord.get (item);

  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::setupPedestalWidths(const CastorPedestalWidthsRcd& fRecord,
                                           CastorDbService* service) {
  edm::ESHandle <CastorPedestalWidths> item;
  fRecord.get (item);
  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Pedestals set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


void CastorDbProducer::setupGains(const CastorGainsRcd& fRecord,
                                  CastorDbService* service) {
  edm::ESHandle <CastorGains> item;
  fRecord.get (item);
  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Gains set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


void CastorDbProducer::setupGainWidths(const CastorGainWidthsRcd& fRecord,
                                       CastorDbService* service) {
  edm::ESHandle <CastorGainWidths> item;
  fRecord.get (item);
  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR GainWidths set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::setupQIEData(const CastorQIEDataRcd& fRecord,
                                    CastorDbService* service) {
  edm::ESHandle <CastorQIEData> item;
  fRecord.get (item);
  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR QIEData set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::setupChannelQuality(const CastorChannelQualityRcd& fRecord,
                                           CastorDbService* service) {
  edm::ESHandle <CastorChannelQuality> item;
  fRecord.get (item);
  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR ChannelQuality set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void CastorDbProducer::setupElectronicsMap(const CastorElectronicsMapRcd& fRecord,
                                           CastorDbService* service) {
  edm::ESHandle <CastorElectronicsMap> item;
  fRecord.get (item);
  service->setData(item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR Electronics Map set" << std::endl;
    CastorDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}
