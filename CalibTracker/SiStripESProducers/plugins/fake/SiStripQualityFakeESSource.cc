// -*- C++ -*-
//
// Package:    SiStripQualityFakeESSource
// Class:      SiStripQualityFakeESSource
//
/**\class SiStripQualityFakeESSource  CalibTracker/SiStripQualityFakeESSource/plugins/fake/SiStripQualityFakeESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 11:46:09 CEST 2007
//
//

#include "CalibTracker/SiStripESProducers/plugins/fake/SiStripQualityFakeESSource.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripQualityFakeESSource::SiStripQualityFakeESSource(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
  findingRecord<SiStripQualityRcd>();
}

std::unique_ptr<SiStripQuality> SiStripQualityFakeESSource::produce(const SiStripQualityRcd& iRecord) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
  return std::make_unique<SiStripQuality>(detInfo);
}

void SiStripQualityFakeESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                const edm::IOVSyncValue& iov,
                                                edm::ValidityInterval& iValidity) {
  edm::ValidityInterval infinity(iov.beginOfTime(), iov.endOfTime());
  iValidity = infinity;
}
