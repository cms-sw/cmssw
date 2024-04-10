// Author: Jan Kašpar

#ifndef CondFormats_DataRecord_CTPPSInterpolatedOpticsRcd_h
#define CondFormats_DataRecord_CTPPSInterpolatedOpticsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"

#include "FWCore/Utilities/interface/mplVector.h"

class CTPPSInterpolatedOpticsRcd
    : public edm::eventsetup::DependentRecordImplementation<
          CTPPSInterpolatedOpticsRcd,
          edm::mpl::Vector<CTPPSOpticsRcd, LHCInfoRcd, LHCInfoPerFillRcd, LHCInfoPerLSRcd>> {};

#endif
