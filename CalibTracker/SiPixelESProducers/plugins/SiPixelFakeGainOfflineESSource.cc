// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelGainESProducers
// Class:      SiPixelFakeGainOfflineESSource
//
/**\class SiPixelFakeGainOfflineESSource SiPixelFakeGainOfflineESSource.cc CalibTracker/SiPixelGainESProducers/plugins/SiPixelFakeGainOfflineESSource.cc

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
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationOfflineRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
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

class SiPixelFakeGainOfflineESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelFakeGainOfflineESSource(const edm::ParameterSet&);
  ~SiPixelFakeGainOfflineESSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual std::unique_ptr<SiPixelGainCalibrationOffline> produce(const SiPixelGainCalibrationOfflineRcd&);

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
SiPixelFakeGainOfflineESSource::SiPixelFakeGainOfflineESSource(const edm::ParameterSet& conf_)
    : fp_(conf_.getParameter<edm::FileInPath>("file")) {
  edm::LogInfo("SiPixelFakeGainOfflineESSource::SiPixelFakeGainOfflineESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGainCalibrationOfflineRcd>();
}

std::unique_ptr<SiPixelGainCalibrationOffline> SiPixelFakeGainOfflineESSource::produce(
    const SiPixelGainCalibrationOfflineRcd&) {
  using namespace edm::es;
  SiPixelGainCalibrationOffline* obj = new SiPixelGainCalibrationOffline(25., 30., 2., 3.);
  SiPixelDetInfoFileReader reader(fp_.fullPath());
  const std::vector<uint32_t>& DetIds = reader.getAllDetIds();

  // Loop over detectors
  for (std::vector<uint32_t>::const_iterator detit = DetIds.begin(); detit != DetIds.end(); detit++) {
    std::vector<char> theSiPixelGainCalibrationOffline;
    const std::pair<int, int>& detUnitDimensions = reader.getDetUnitDimensions(*detit);

    // Loop over columns and rows
    for (int i = 0; i < detUnitDimensions.first; i++) {
      float totalGain = 0.0;
      float totalEntries = 0.0;
      for (int j = 0; j < detUnitDimensions.second; j++) {
        totalGain += 2.8;
        float ped = 28.2;
        totalEntries += 1.0;
        obj->setDataPedestal(ped, theSiPixelGainCalibrationOffline);
        if ((j + 1) % 80 == 0)  //compute the gain average after each ROC
        {
          float gain = totalGain / totalEntries;
          obj->setDataGain(gain, 80, theSiPixelGainCalibrationOffline);
          totalGain = 0;
          totalEntries = 0.0;
        }
      }
    }

    //std::cout << "detid " << (*detit) << std::endl;

    SiPixelGainCalibrationOffline::Range range(theSiPixelGainCalibrationOffline.begin(),
                                               theSiPixelGainCalibrationOffline.end());
    // the 80 in the line below represents the number of columns averaged over.
    if (!obj->put(*detit, range, detUnitDimensions.first))
      edm::LogError("SiPixelFakeGainOfflineESSource")
          << "[SiPixelFakeGainOfflineESSource::produce] detid already exists" << std::endl;
  }

  //
  return std::unique_ptr<SiPixelGainCalibrationOffline>(obj);
}

void SiPixelFakeGainOfflineESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                    const edm::IOVSyncValue& iosv,
                                                    edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelFakeGainOfflineESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainOfflineESSource);
