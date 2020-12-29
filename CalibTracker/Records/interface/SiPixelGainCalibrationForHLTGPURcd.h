#ifndef CalibTracker_Records_SiPixelGainCalibrationForHLTGPURcd_h
#define CalibTracker_Records_SiPixelGainCalibrationForHLTGPURcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "boost/mpl/vector.hpp"

class SiPixelGainCalibrationForHLTGPURcd : public edm::eventsetup::DependentRecordImplementation<SiPixelGainCalibrationForHLTGPURcd, boost::mpl::vector<SiPixelGainCalibrationForHLTRcd, TrackerDigiGeometryRecord> > {};

#endif
