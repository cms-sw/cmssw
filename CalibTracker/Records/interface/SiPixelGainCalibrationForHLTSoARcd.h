#ifndef CalibTracker_Records_SiPixelGainCalibrationForHLTSoARcd_h
#define CalibTracker_Records_SiPixelGainCalibrationForHLTSoARcd_h

#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class SiPixelGainCalibrationForHLTSoARcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixelGainCalibrationForHLTSoARcd,
          edm::mpl::Vector<SiPixelGainCalibrationForHLTRcd, TrackerDigiGeometryRecord>> {};

#endif  // CalibTracker_Records_SiPixelGainCalibrationForHLTSoARcd_h
