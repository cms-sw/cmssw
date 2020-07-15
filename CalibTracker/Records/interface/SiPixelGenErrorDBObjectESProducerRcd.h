#ifndef CalibTracker_Records_SiPixelGenErrorDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixelGenErrorDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <boost/mp11/list.hpp>

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

class SiPixelGenErrorDBObjectESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<
          SiPixelGenErrorDBObjectESProducerRcd,
          boost::mp11::mp_list<IdealMagneticFieldRecord, SiPixelGenErrorDBObjectRcd> > {};

#endif
