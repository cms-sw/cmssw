#ifndef CondFormats_DataRecord_interface_HcalCombinedRecordsGPU_h
#define CondFormats_DataRecord_interface_HcalCombinedRecordsGPU_h

#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

template <typename... Sources>
class HcalCombinedRecord : public edm::eventsetup::DependentRecordImplementation<HcalCombinedRecord<Sources...>,
                                                                                 edm::mpl::Vector<Sources...>> {};

using HcalConvertedPedestalsRcd = HcalCombinedRecord<HcalPedestalsRcd, HcalQIEDataRcd, HcalQIETypesRcd>;

using HcalConvertedPedestalWidthsRcd =
    HcalCombinedRecord<HcalPedestalsRcd, HcalPedestalWidthsRcd, HcalQIEDataRcd, HcalQIETypesRcd>;

#endif  // CondFormats_DataRecord_interface_HcalCombinedRecordsGPU_h
