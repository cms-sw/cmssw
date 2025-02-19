#ifndef MagneticField_IdealMagneticFieldRecord_h
#define MagneticField_IdealMagneticFieldRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "boost/mpl/vector.hpp"

// class IdealMagneticFieldRecord : public edm::eventsetup::EventSetupRecordImplementation<IdealMagneticFieldRecord> {};

class IdealMagneticFieldRecord : public edm::eventsetup::DependentRecordImplementation<IdealMagneticFieldRecord,boost::mpl::vector<RunInfoRcd> > {};

#endif

