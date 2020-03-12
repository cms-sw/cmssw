// Author: Jan Kašpar

#ifndef CondFormats_DataRecord_CTPPSInterpolatedOpticsRcd_h
#define CondFormats_DataRecord_CTPPSInterpolatedOpticsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "boost/mpl/vector.hpp"

class CTPPSInterpolatedOpticsRcd
    : public edm::eventsetup::DependentRecordImplementation<CTPPSInterpolatedOpticsRcd,
                                                            boost::mpl::vector<CTPPSOpticsRcd, LHCInfoRcd>> {};

#endif
