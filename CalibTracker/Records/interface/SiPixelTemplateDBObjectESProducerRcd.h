#ifndef CalibTracker_Records_SiPixelTemplateDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixelTemplateDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

class SiPixelTemplateDBObjectESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixelTemplateDBObjectESProducerRcd,
          edm::mpl::Vector<IdealMagneticFieldRecord, SiPixelTemplateDBObjectRcd> > {};

#endif
