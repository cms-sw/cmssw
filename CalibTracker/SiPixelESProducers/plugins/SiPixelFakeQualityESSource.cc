// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelESProducers
// Class:      SiPixelFakeQualityESSource
//
/**\class CalibTracker/SiPixelESProducers/plugins/SiPixelFakeQualityESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bernadette Heyburn
//         Created:  Oct 21 2008
//
//

// system include files
#include <memory>

// user include files
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//
// class decleration
//

class SiPixelFakeQualityESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelFakeQualityESSource(const edm::ParameterSet&);
  ~SiPixelFakeQualityESSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual std::unique_ptr<SiPixelQuality> produce(const SiPixelQualityFromDbRcd&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  edm::FileInPath fp_;
};

//
// constructors and destructor
//
SiPixelFakeQualityESSource::SiPixelFakeQualityESSource(const edm::ParameterSet& conf_)
    : fp_(conf_.getParameter<edm::FileInPath>("file")) {
  edm::LogInfo("SiPixelFakeQualityESSource::SiPixelFakeQualityESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelQualityFromDbRcd>();
}

std::unique_ptr<SiPixelQuality> SiPixelFakeQualityESSource::produce(const SiPixelQualityFromDbRcd&) {
  ///////////////////////////////////////////////////////
  //  errortype "whole" = int 0 in DB  BadRocs = 65535 //
  //  errortype "tbmA" = int 1 in DB  BadRocs = 255    //
  //  errortype "tbmB" = int 2 in DB  Bad Rocs = 65280 //
  //  errortype "none" = int 3 in DB                   //
  ///////////////////////////////////////////////////////

  SiPixelQuality* obj = new SiPixelQuality();

  SiPixelQuality::disabledModuleType BadModule;
  BadModule.DetID = 1;
  BadModule.errorType = 0;
  BadModule.BadRocs = 65535;
  obj->addDisabledModule(BadModule);

  return std::unique_ptr<SiPixelQuality>(obj);
}

void SiPixelFakeQualityESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                const edm::IOVSyncValue& iosv,
                                                edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelFakeQualityESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeQualityESSource);
