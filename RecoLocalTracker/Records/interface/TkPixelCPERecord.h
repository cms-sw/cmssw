#ifndef RecoLocalTracker_Records_TkPixelCPERecord_h
#define RecoLocalTracker_Records_TkPixelCPERecord_h

#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

#include "boost/mpl/vector.hpp"

class TkPixelCPERecord
    : public edm::eventsetup::DependentRecordImplementation<
          TkPixelCPERecord,
          boost::mpl::vector<TrackerDigiGeometryRecord,
                             IdealMagneticFieldRecord, SiPixelLorentzAngleRcd,
                             SiPixelGenErrorDBObjectRcd,
                             SiPixelTemplateDBObjectESProducerRcd,
                             SiPixel2DTemplateDBObjectESProducerRcd>> {};

#endif
