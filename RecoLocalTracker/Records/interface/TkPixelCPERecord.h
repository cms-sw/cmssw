#ifndef RecoLocalTracker_Records_TkPixelCPERecord_h
#define RecoLocalTracker_Records_TkPixelCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
//#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TkPixelCPERecord
    : public edm::eventsetup::DependentRecordImplementation<TkPixelCPERecord,
                                                            edm::mpl::Vector<TrackerDigiGeometryRecord,
                                                                             IdealMagneticFieldRecord,
                                                                             SiPixelLorentzAngleRcd,
                                                                             SiPixelGenErrorDBObjectRcd,
                                                                             SiPixelTemplateDBObjectESProducerRcd,
                                                                             SiPixel2DTemplateDBObjectESProducerRcd,
                                                                             TrackerTopologyRcd> > {};

#endif
