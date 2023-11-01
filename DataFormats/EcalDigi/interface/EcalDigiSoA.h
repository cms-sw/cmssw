#ifndef DataFormats_EcalDigi_EcalDigiSoA_h
#define DataFormats_EcalDigi_EcalDigiSoA_h

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// due to a ROOT limitation the std::array needs to be wrapped
// https://github.com/root-project/root/issues/12007
using EcalDataArray = edm::StdArray<uint16_t, ecalPh1::sampleSize>;

GENERATE_SOA_LAYOUT(EcalDigiSoALayout,
                    SOA_COLUMN(uint32_t, id),
                    SOA_COLUMN(EcalDataArray, data),
                    SOA_SCALAR(uint32_t, size))

using EcalDigiSoA = EcalDigiSoALayout<>;

#endif
