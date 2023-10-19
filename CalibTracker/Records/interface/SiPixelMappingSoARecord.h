#ifndef CalibTracker_Records_interface_SiPixelMappingSoARecord_h
#define CalibTracker_Records_interface_SiPixelMappingSoARecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"

class SiPixelMappingSoARecord
    : public edm::eventsetup::DependentRecordImplementation<SiPixelMappingSoARecord,
                                                            edm::mpl::Vector<SiPixelGainCalibrationForHLTRcd,
                                                                             SiPixelQualityRcd,
                                                                             SiPixelFedCablingMapRcd,
                                                                             TrackerDigiGeometryRecord>> {};

#endif