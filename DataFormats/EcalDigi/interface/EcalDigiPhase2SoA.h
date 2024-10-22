#ifndef DataFormats_EcalDigi_EcalDigiPhase2SoA_h
#define DataFormats_EcalDigi_EcalDigiPhase2SoA_h

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// due to a ROOT limitation the std::array needs to be wrapped
// https://github.com/root-project/root/issues/12007
using EcalDataArrayPhase2 = edm::StdArray<uint16_t, ecalPh2::sampleSize>;

GENERATE_SOA_LAYOUT(EcalDigiPhase2SoALayout,
                    SOA_COLUMN(uint32_t, id),
                    SOA_COLUMN(EcalDataArrayPhase2, data),
                    SOA_SCALAR(uint32_t, size))

using EcalDigiPhase2SoA = EcalDigiPhase2SoALayout<>;

#endif
