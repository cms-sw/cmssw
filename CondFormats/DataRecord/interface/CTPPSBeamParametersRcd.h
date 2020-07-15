// Authors:
//  Wagner De Paula Carvalho
//  Jan Kašpar

#ifndef CTPPSBeamParametersRcd_CTPPSBeamParametersRcd_h
#define CTPPSBeamParametersRcd_CTPPSBeamParametersRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include <boost/mp11/list.hpp>

class CTPPSBeamParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<CTPPSBeamParametersRcd, boost::mp11::mp_list<LHCInfoRcd>> {};

#endif
