#ifndef CalibTracker_Records_SiPixelGenErrorDBObjectESProducerRcd_h
#define CalibTracker_Records_SiPixelGenErrorDBObjectESProducerRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

class SiPixelGenErrorDBObjectESProducerRcd : public edm::eventsetup::DependentRecordImplementation<SiPixelGenErrorDBObjectESProducerRcd, boost::mpl::vector<IdealMagneticFieldRecord, SiPixelGenErrorDBObjectRcd> > {};

#endif
