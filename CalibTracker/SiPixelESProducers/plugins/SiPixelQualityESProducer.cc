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

#include <memory>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

class SiPixelQualityESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelQualityESProducer(const edm::ParameterSet& iConfig);
  ~SiPixelQualityESProducer() override = default;

  std::unique_ptr<SiPixelQuality> produce(const SiPixelQualityRcd& iRecord);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

  edm::ESGetToken<SiPixelQuality, SiPixelQualityFromDbRcd> siPixelQualityFromDbToken_;
  edm::ESGetToken<SiStripDetVOff, SiPixelDetVOffRcd> voffToken_;
};

SiPixelQualityESProducer::SiPixelQualityESProducer(const edm::ParameterSet& iConfig) {
  // setWhatProduced internally uses "appendToDataLabel" to name the output product of this ESProducer
  auto const& appendToDataLabel = iConfig.getParameter<std::string>("appendToDataLabel");
  auto esCC = setWhatProduced(this);

  // "siPixelQualityFromDbLabel" corresponds to the Label of a tag with Record
  // "SiPixelQualityFromDbRcd" in the EventSetup (normally provided by the GlobalTag)
  auto const& siPixelQualityFromDbLabel = iConfig.getParameter<std::string>("siPixelQualityFromDbLabel");
  siPixelQualityFromDbToken_ = esCC.consumes(edm::ESInputTag{"", siPixelQualityFromDbLabel});
  voffToken_ = esCC.consumes();

  findingRecord<SiPixelQualityRcd>();

  edm::LogInfo("SiPixelQualityESProducer")
      << "Module = \"" << description().label_ << "\" (appendToDataLabel = \"" << appendToDataLabel
      << "\", siPixelQualityFromDbLabel = \"" << siPixelQualityFromDbLabel << "\")";
}

std::unique_ptr<SiPixelQuality> SiPixelQualityESProducer::produce(const SiPixelQualityRcd& iRecord) {
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
  auto dbptr = std::make_unique<SiPixelQuality>(iRecord.get(siPixelQualityFromDbToken_));

  //here is the magic line in which it switches off Bad Modules
  dbptr->add(&(iRecord.get(voffToken_)));
  return dbptr;
}

void SiPixelQualityESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                              const edm::IOVSyncValue& iosv,
                                              edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelQualityESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("siPixelQualityFromDbLabel", "");
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
