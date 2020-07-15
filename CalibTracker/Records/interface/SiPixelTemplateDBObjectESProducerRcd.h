#ifndef CalibTracker_Records_SiPixelTemplateDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixelTemplateDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <boost/mp11/list.hpp>

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

class SiPixelTemplateDBObjectESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixelTemplateDBObjectESProducerRcd,
          boost::mp11::mp_list<IdealMagneticFieldRecord, SiPixelTemplateDBObjectRcd> > {};

#endif
