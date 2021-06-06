#ifndef CalibTracker_Records_SiPixelGainCalibrationForHLTGPURcd_h
#define CalibTracker_Records_SiPixelGainCalibrationForHLTGPURcd_h

#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class SiPixelGainCalibrationForHLTGPURcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixelGainCalibrationForHLTGPURcd,
          edm::mpl::Vector<SiPixelGainCalibrationForHLTRcd, TrackerDigiGeometryRecord>> {};

#endif  // CalibTracker_Records_SiPixelGainCalibrationForHLTGPURcd_h
