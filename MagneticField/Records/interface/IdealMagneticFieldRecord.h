#ifndef MagneticField_IdealMagneticFieldRecord_h
#define MagneticField_IdealMagneticFieldRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class IdealMagneticFieldRecord : public edm::eventsetup::DependentRecordImplementation<IdealMagneticFieldRecord,
 boost::mpl::vector<IdealGeometryRecord> > {};

#endif

