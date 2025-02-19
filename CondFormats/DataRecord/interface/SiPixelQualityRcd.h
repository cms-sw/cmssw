#ifndef DataRecord_SiPixelQualityRcd_h
#define DataRecord_SiPixelQualityRcd_h


#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"


#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
//#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

class SiPixelDetVOffRcd : public edm::eventsetup::EventSetupRecordImplementation<SiPixelDetVOffRcd> {};

class SiPixelQualityRcd : public edm::eventsetup::DependentRecordImplementation<SiPixelQualityRcd, boost::mpl::vector<SiPixelQualityFromDbRcd, SiPixelDetVOffRcd> > {};

#endif
