// Author: Jan Kašpar

#ifndef CondFormats_DataRecord_CTPPSInterpolatedOpticsRcd_h
#define CondFormats_DataRecord_CTPPSInterpolatedOpticsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include <boost/mp11/list.hpp>

class CTPPSInterpolatedOpticsRcd
    : public edm::eventsetup::DependentRecordImplementation<CTPPSInterpolatedOpticsRcd,
                                                            boost::mp11::mp_list<CTPPSOpticsRcd, LHCInfoRcd>> {};

#endif
