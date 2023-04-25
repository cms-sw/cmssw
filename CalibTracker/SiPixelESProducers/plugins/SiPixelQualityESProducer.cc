// -*- C++ -*-
//
// Package:    SiPixelQualityESProducer
// Class:      SiPixelQualityESProducer
//
/**\class SiPixelQualityESProducer SiPixelQualityESProducer.h CalibTracker/SiPixelESProducer/src/SiPixelQualityESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Gemma Tinti
//         Created:  Jan 13 2011
//
//

// system include files
#include <memory>
#include <cassert>

// user include files
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

using namespace edm;

class SiPixelQualityESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelQualityESProducer(const edm::ParameterSet& iConfig);
  ~SiPixelQualityESProducer() override;

  std::unique_ptr<SiPixelQuality> produce(const SiPixelQualityRcd& iRecord);
  std::unique_ptr<SiPixelQuality> produceWithLabel(const SiPixelQualityRcd& iRecord);
  std::unique_ptr<SiPixelQuality> produceWithLabelRawToDigi(const SiPixelQualityRcd& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

  struct Tokens {
    Tokens() = default;
    explicit Tokens(edm::ESConsumesCollector cc, const std::string& label) {
      voffToken_ = cc.consumes();
      dbobjectToken_ = cc.consumes(edm::ESInputTag{"", label});
    }
    edm::ESGetToken<SiStripDetVOff, SiPixelDetVOffRcd> voffToken_;
    edm::ESGetToken<SiPixelQuality, SiPixelQualityFromDbRcd> dbobjectToken_;
  };

  std::unique_ptr<SiPixelQuality> get_pointer(const SiPixelQualityRcd& iRecord, const Tokens& tokens);

  const Tokens defaultTokens_;
  Tokens labelTokens_;
  Tokens labelTokens_RawToDigi_;
};

//
// constructors and destructor
//

SiPixelQualityESProducer::SiPixelQualityESProducer(const edm::ParameterSet& conf_)
    : defaultTokens_(setWhatProduced(this), "") {
  edm::LogInfo("SiPixelQualityESProducer::SiPixelQualityESProducer");

  auto label = conf_.getParameter<std::string>("siPixelQualityLabel");

  if (label == "forDigitizer") {
    labelTokens_ =
        Tokens(setWhatProduced(this, &SiPixelQualityESProducer::produceWithLabel, edm::es::Label(label)), label);
  }

  label = conf_.getParameter<std::string>("siPixelQualityLabel_RawToDigi");

  if (label == "forRawToDigi") {
    labelTokens_RawToDigi_ = Tokens(
        setWhatProduced(this, &SiPixelQualityESProducer::produceWithLabelRawToDigi, edm::es::Label(label)), label);
  }
  findingRecord<SiPixelQualityRcd>();
}

SiPixelQualityESProducer::~SiPixelQualityESProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

std::unique_ptr<SiPixelQuality> SiPixelQualityESProducer::get_pointer(const SiPixelQualityRcd& iRecord,
                                                                      const Tokens& tokens) {
  ///////////////////////////////////////////////////////
  //  errortype "whole" = int 0 in DB  BadRocs = 65535 //
  //  errortype "tbmA" = int 1 in DB  BadRocs = 255    //
  //  errortype "tbmB" = int 2 in DB  Bad Rocs = 65280 //
  //  errortype "none" = int 3 in DB                   //
  ///////////////////////////////////////////////////////

  //if I have understood this is the one got from the DB or file, but in any case the ORIGINAL(maybe i need to get the record for it)
  //SiPixelQuality * obj = new SiPixelQuality();
  //SiPixelQuality::disabledModuleType BadModule;
  //BadModule.DetID = 1; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);

  //now the dbobject is the one copied from the db
  auto dbptr = std::make_unique<SiPixelQuality>(iRecord.get(tokens.dbobjectToken_));

  //here is the magic line in which it switches off Bad Modules
  dbptr->add(&(iRecord.get(tokens.voffToken_)));
  return dbptr;
}

std::unique_ptr<SiPixelQuality> SiPixelQualityESProducer::produce(const SiPixelQualityRcd& iRecord) {
  return get_pointer(iRecord, defaultTokens_);
}
std::unique_ptr<SiPixelQuality> SiPixelQualityESProducer::produceWithLabel(const SiPixelQualityRcd& iRecord) {
  return get_pointer(iRecord, labelTokens_);
}
std::unique_ptr<SiPixelQuality> SiPixelQualityESProducer::produceWithLabelRawToDigi(const SiPixelQualityRcd& iRecord) {
  return get_pointer(iRecord, labelTokens_RawToDigi_);
}

void SiPixelQualityESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                              const edm::IOVSyncValue& iosv,
                                              edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelQualityESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("siPixelQualityLabel", "");
  desc.add<std::string>("siPixelQualityLabel_RawToDigi", "");
  {
    edm::ParameterSetDescription desc_ps;
    desc_ps.add<std::string>("record", "SiPixelQualityFromDbRcd");
    desc_ps.add<std::string>("tag", "");
    std::vector<edm::ParameterSet> default_ps;
    default_ps.reserve(2);
    {
      edm::ParameterSet temp;
      temp.addParameter<std::string>("record", "SiPixelQualityFromDbRcd");
      temp.addParameter<std::string>("tag", "");
      default_ps.push_back(temp);
    }
    {
      edm::ParameterSet temp;
      temp.addParameter<std::string>("record", "SiPixelDetVOffRcd");
      temp.addParameter<std::string>("tag", "");
      default_ps.push_back(temp);
    }
    desc.addVPSet("ListOfRecordToMerge", desc_ps, default_ps);
  }
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelQualityESProducer);
