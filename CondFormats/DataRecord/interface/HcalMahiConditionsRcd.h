#ifndef CondFormats_DataRecord_HcalMahiConditionsRcd_h
#define CondFormats_DataRecord_HcalMahiConditionsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMParametersRcd.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalMahiConditionsRcd
    : public edm::eventsetup::DependentRecordImplementation<HcalMahiConditionsRcd,
                                                            edm::mpl::Vector<HcalRecoParamsRcd,
                                                                             HcalPedestalsRcd,
                                                                             HcalGainsRcd,
                                                                             HcalLUTCorrsRcd,
                                                                             HcalRespCorrsRcd,
                                                                             HcalTimeCorrsRcd,
                                                                             HcalPedestalWidthsRcd,
                                                                             HcalGainWidthsRcd,
                                                                             HcalChannelQualityRcd,
                                                                             HcalQIETypesRcd,
                                                                             HcalQIEDataRcd,
                                                                             HcalSiPMParametersRcd,
                                                                             HcalRecNumberingRecord>> {};
#endif
