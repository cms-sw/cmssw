#ifndef CalibTracker_Records_SiPixel2DTemplateDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixel2DTemplateDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class SiPixel2DTemplateDBObjectESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixel2DTemplateDBObjectESProducerRcd,
          edm::mpl::Vector<IdealMagneticFieldRecord, SiPixel2DTemplateDBObjectRcd> > {};

#endif
