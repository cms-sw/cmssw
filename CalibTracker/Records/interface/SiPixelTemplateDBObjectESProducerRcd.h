#ifndef CalibTracker_Records_SiPixelTemplateDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixelTemplateDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject0TRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject38TRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject4TRcd.h"

class SiPixelTemplateDBObjectESProducerRcd : public edm::eventsetup::DependentRecordImplementation<SiPixelTemplateDBObjectESProducerRcd, boost::mpl::vector<IdealMagneticFieldRecord, SiPixelTemplateDBObject0TRcd, SiPixelTemplateDBObject38TRcd, SiPixelTemplateDBObject4TRcd> > {};

#endif
