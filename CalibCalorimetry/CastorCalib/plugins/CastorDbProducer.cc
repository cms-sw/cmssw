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
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

class CastorDbProducer : public edm::ESProducer {
public:
  CastorDbProducer(const edm::ParameterSet&);
  ~CastorDbProducer() override;

  std::shared_ptr<CastorDbService> produce(const CastorDbRecord&);

private:
  // ----------member data ---------------------------
  using HostType = edm::ESProductHost<CastorDbService,
                                      CastorPedestalsRcd,
                                      CastorPedestalWidthsRcd,
                                      CastorGainsRcd,
                                      CastorGainWidthsRcd,
                                      CastorQIEDataRcd,
                                      CastorChannelQualityRcd,
                                      CastorElectronicsMapRcd>;

  template <typename RecordT, typename TokenT>
  void setupItem(const RecordT& fRecord, const TokenT& token, const char* name, CastorDbService* service);

  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<CastorPedestals, CastorPedestalsRcd> pedestalsToken_;
  edm::ESGetToken<CastorPedestalWidths, CastorPedestalWidthsRcd> pedestalWidthsToken_;
  edm::ESGetToken<CastorGains, CastorGainsRcd> gainsToken_;
  edm::ESGetToken<CastorGainWidths, CastorGainWidthsRcd> gainWidthsToken_;
  edm::ESGetToken<CastorQIEData, CastorQIEDataRcd> qieDataToken_;
  edm::ESGetToken<CastorChannelQuality, CastorChannelQualityRcd> channelQualityToken_;
  edm::ESGetToken<CastorElectronicsMap, CastorElectronicsMapRcd> electronicsMapToken_;

  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;
};

CastorDbProducer::CastorDbProducer(const edm::ParameterSet& fConfig)
    : ESProducer(), mDumpRequest(), mDumpStream(nullptr) {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this);
  pedestalsToken_ = cc.consumes();
  pedestalWidthsToken_ = cc.consumes();
  gainsToken_ = cc.consumes();
  gainWidthsToken_ = cc.consumes();
  qieDataToken_ = cc.consumes();
  channelQualityToken_ = cc.consumes();
  electronicsMapToken_ = cc.consumes();

  //now do what ever other initialization is needed

  mDumpRequest = fConfig.getUntrackedParameter<std::vector<std::string> >("dump", std::vector<std::string>());
  if (!mDumpRequest.empty()) {
    std::string otputFile = fConfig.getUntrackedParameter<std::string>("file", "");
    mDumpStream = otputFile.empty() ? &std::cout : new std::ofstream(otputFile.c_str());
  }
}

CastorDbProducer::~CastorDbProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (mDumpStream != &std::cout)
    delete mDumpStream;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
std::shared_ptr<CastorDbService> CastorDbProducer::produce(const CastorDbRecord& record) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  bool needBuildCalibrations = false;
  bool needBuildCalibWidths = false;

  host->ifRecordChanges<CastorElectronicsMapRcd>(
      record, [this, h = host.get()](auto const& rec) { setupItem(rec, electronicsMapToken_, "ElectronicsMap", h); });
  host->ifRecordChanges<CastorChannelQualityRcd>(
      record, [this, h = host.get()](auto const& rec) { setupItem(rec, channelQualityToken_, "ChannelQuality", h); });
  host->ifRecordChanges<CastorGainWidthsRcd>(record, [this, h = host.get(), &needBuildCalibWidths](auto const& rec) {
    setupItem(rec, gainWidthsToken_, "GainWidths", h);
    needBuildCalibWidths = true;
  });
  host->ifRecordChanges<CastorQIEDataRcd>(
      record, [this, h = host.get(), &needBuildCalibrations, &needBuildCalibWidths](auto const& rec) {
        setupItem(rec, qieDataToken_, "QIEData", h);
        needBuildCalibrations = true;
        needBuildCalibWidths = true;
      });
  host->ifRecordChanges<CastorPedestalWidthsRcd>(record,
                                                 [this, h = host.get(), &needBuildCalibWidths](auto const& rec) {
                                                   setupItem(rec, pedestalWidthsToken_, "PedestalWidths", h);
                                                   needBuildCalibWidths = true;
                                                 });
  host->ifRecordChanges<CastorGainsRcd>(record, [this, h = host.get(), &needBuildCalibrations](auto const& rec) {
    setupItem(rec, gainsToken_, "Gains", h);
    needBuildCalibrations = true;
  });
  host->ifRecordChanges<CastorPedestalsRcd>(record, [this, h = host.get(), &needBuildCalibrations](auto const& rec) {
    setupItem(rec, pedestalsToken_, "Pedestals", h);
    needBuildCalibrations = true;
  });

  if (needBuildCalibWidths) {
    host->buildCalibWidths();
  }

  if (needBuildCalibrations) {
    host->buildCalibrations();
  }

  return host;  // automatically converts to std::shared_ptr<CastorDbService>
}

template <typename RecordT, typename TokenT>
void CastorDbProducer::setupItem(const RecordT& fRecord,
                                 const TokenT& token,
                                 const char* name,
                                 CastorDbService* service) {
  const auto& item = fRecord.get(token);
  service->setData(&item);
  if (std::find(mDumpRequest.begin(), mDumpRequest.end(), name) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL/CASTOR " << name << " set" << std::endl;
    CastorDbASCIIIO::dumpObject(*mDumpStream, item);
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(CastorDbProducer);
