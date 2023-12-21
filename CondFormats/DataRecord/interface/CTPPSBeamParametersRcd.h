// Authors:
//  Wagner De Paula Carvalho
//  Jan Kašpar

#ifndef CTPPSBeamParametersRcd_CTPPSBeamParametersRcd_h
#define CTPPSBeamParametersRcd_CTPPSBeamParametersRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class CTPPSBeamParametersRcd : public edm::eventsetup::DependentRecordImplementation<
                                   CTPPSBeamParametersRcd,
                                   edm::mpl::Vector<LHCInfoRcd, LHCInfoPerFillRcd, LHCInfoPerLSRcd>> {};

#endif
