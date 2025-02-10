// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelGainForHLTESProducers
// Class:      SiPixelFakeGainForHLTESSource
//
/**\class SiPixelFakeGainForHLTESSource SiPixelFakeGainForHLTESSource.cc CalibTracker/SiPixelGainForHLTESProducers/plugins/SiPixelFakeGainForHLTESSource.cc

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
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
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

class SiPixelFakeGainForHLTESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelFakeGainForHLTESSource(const edm::ParameterSet&);
  ~SiPixelFakeGainForHLTESSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual std::unique_ptr<SiPixelGainCalibrationForHLT> produce(const SiPixelGainCalibrationForHLTRcd&);

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
SiPixelFakeGainForHLTESSource::SiPixelFakeGainForHLTESSource(const edm::ParameterSet& conf_)
    : fp_(conf_.getParameter<edm::FileInPath>("file")) {
  edm::LogInfo("SiPixelFakeGainForHLTESSource::SiPixelFakeGainForHLTESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGainCalibrationForHLTRcd>();
}

std::unique_ptr<SiPixelGainCalibrationForHLT> SiPixelFakeGainForHLTESSource::produce(
    const SiPixelGainCalibrationForHLTRcd&) {
  using namespace edm::es;
  SiPixelGainCalibrationForHLT* obj = new SiPixelGainCalibrationForHLT(25., 30., 2., 3.);
  SiPixelDetInfoFileReader reader(fp_.fullPath());
  const std::vector<uint32_t>& DetIds = reader.getAllDetIds();

  // Loop over detectors
  for (std::vector<uint32_t>::const_iterator detit = DetIds.begin(); detit != DetIds.end(); detit++) {
    std::vector<char> theSiPixelGainCalibration;
    const std::pair<int, int>& detUnitDimensions = reader.getDetUnitDimensions(*detit);

    // Loop over columns and rows

    for (int i = 0; i < detUnitDimensions.first; i++) {
      float totalGain = 0.0;
      float totalPed = 0.0;
      float totalEntries = 0.0;
      for (int j = 0; j < detUnitDimensions.second; j++) {
        totalGain += 2.8;
        totalPed += 28.2;
        totalEntries++;
        if ((j + 1) % 80 == 0) {
          float gain = totalGain / totalEntries;
          float ped = totalPed / totalEntries;

          obj->setData(ped, gain, theSiPixelGainCalibration);
          totalGain = 0.;
          totalPed = 0.;
          totalEntries = 0.;
        }
      }
    }

    //std::cout << "detid " << (*detit) << std::endl;

    SiPixelGainCalibrationForHLT::Range range(theSiPixelGainCalibration.begin(), theSiPixelGainCalibration.end());
    int nCols = detUnitDimensions.first;
    if (!obj->put(*detit, range, nCols))
      edm::LogError("SiPixelFakeGainForHLTESSource")
          << "[SiPixelFakeGainForHLTESSource::produce] detid already exists" << std::endl;
  }

  //
  return std::unique_ptr<SiPixelGainCalibrationForHLT>(obj);
}

void SiPixelFakeGainForHLTESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                   const edm::IOVSyncValue& iosv,
                                                   edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelFakeGainForHLTESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGainForHLTESSource);
