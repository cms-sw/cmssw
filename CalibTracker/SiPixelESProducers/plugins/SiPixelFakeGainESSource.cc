// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelESProducers
// Class:      SiPixelFakeGainESSource
//
/**\class SiPixelFakeGainESSource SiPixelFakeGainESSource.cc CalibTracker/SiPixelGainESProducer/plugins/SiPixelFakeGainESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Tue 8 12:31:25 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
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

class SiPixelFakeGainESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelFakeGainESSource(const edm::ParameterSet&);
  ~SiPixelFakeGainESSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual std::unique_ptr<SiPixelGainCalibration> produce(const SiPixelGainCalibrationRcd&);

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
SiPixelFakeGainESSource::SiPixelFakeGainESSource(const edm::ParameterSet& conf_)
    : fp_(conf_.getParameter<edm::FileInPath>("file")) {
  edm::LogInfo("SiPixelFakeGainESSource::SiPixelFakeGainESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGainCalibrationRcd>();
}

std::unique_ptr<SiPixelGainCalibration> SiPixelFakeGainESSource::produce(const SiPixelGainCalibrationRcd&) {
  using namespace edm::es;
  SiPixelGainCalibration* obj = new SiPixelGainCalibration(25., 30., 2., 3.);
  SiPixelDetInfoFileReader reader(fp_.fullPath());
  const std::vector<uint32_t>& DetIds = reader.getAllDetIds();

  // Loop over detectors
  for (std::vector<uint32_t>::const_iterator detit = DetIds.begin(); detit != DetIds.end(); detit++) {
    std::vector<char> theSiPixelGainCalibration;
    const std::pair<int, int>& detUnitDimensions = reader.getDetUnitDimensions(*detit);

    // Loop over columns and rows
    for (int i = 0; i < detUnitDimensions.first; i++) {
      for (int j = 0; j < detUnitDimensions.second; j++) {
        float gain = 2.8;
        float ped = 28.2;
        obj->setData(ped, gain, theSiPixelGainCalibration);
      }
    }

    //std::cout << "detid " << (*detit) << std::endl;

    SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(), theSiPixelGainCalibration.end());
    if (!obj->put(*detit, range, detUnitDimensions.first))
      edm::LogError("SiPixelFakeGainESSource")
          << "[SiPixelFakeGainESSource::produce] detid already exists" << std::endl;
  }

  //
  return std::unique_ptr<SiPixelGainCalibration>(obj);
}

void SiPixelFakeGainESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                             const edm::IOVSyncValue& iosv,
                                             edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelFakeGainESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainESSource);
