#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalCombinedRecordsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalCombinedRecordsGPU_h

#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCombinedRecord.h"

using HcalConvertedPedestalsRcd = HcalCombinedRecord<HcalPedestalsRcd, HcalQIEDataRcd, HcalQIETypesRcd>;

using HcalConvertedEffectivePedestalsRcd = HcalCombinedRecord<HcalPedestalsRcd, HcalQIEDataRcd, HcalQIETypesRcd>;

using HcalConvertedPedestalWidthsRcd =
    HcalCombinedRecord<HcalPedestalsRcd, HcalPedestalWidthsRcd, HcalQIEDataRcd, HcalQIETypesRcd>;

using HcalConvertedEffectivePedestalWidthsRcd =
    HcalCombinedRecord<HcalPedestalsRcd, HcalPedestalWidthsRcd, HcalQIEDataRcd, HcalQIETypesRcd>;

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalCombinedRecordsGPU_h
