// Authors:
//  Wagner De Paula Carvalho
//  Jan Kašpar

#ifndef CTPPSBeamParametersRcd_CTPPSBeamParametersRcd_h
#define CTPPSBeamParametersRcd_CTPPSBeamParametersRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "boost/mpl/vector.hpp"

class CTPPSBeamParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<CTPPSBeamParametersRcd, boost::mpl::vector<LHCInfoRcd>> {};

#endif
