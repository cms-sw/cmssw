#ifndef CalibTracker_Records_SiPixel2DTemplateDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixel2DTemplateDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"

#include "boost/mpl/vector.hpp"

class SiPixel2DTemplateDBObjectESProducerRcd : public edm::eventsetup::DependentRecordImplementation<SiPixel2DTemplateDBObjectESProducerRcd, boost::mpl::vector<IdealMagneticFieldRecord, SiPixel2DTemplateDBObjectRcd> > {};

#endif
