#ifndef DataFormats_HGCalReco_MtdSoA_h
#define DataFormats_HGCalReco_MtdSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

GENERATE_SOA_LAYOUT(MtdSoALayout,
                    SOA_COLUMN(int32_t, trackAsocMTD),
                    SOA_COLUMN(float, time0),
                    SOA_COLUMN(float, time0Err),
                    SOA_COLUMN(float, time),
                    SOA_COLUMN(float, timeErr),
                    SOA_COLUMN(float, MVAquality),
                    SOA_COLUMN(float, pathLength),
                    SOA_COLUMN(float, beta),
                    SOA_COLUMN(float, posInMTD_x),
                    SOA_COLUMN(float, posInMTD_y),
                    SOA_COLUMN(float, posInMTD_z),
                    SOA_COLUMN(float, momentumWithMTD),
                    SOA_COLUMN(float, probPi),
                    SOA_COLUMN(float, probK),
                    SOA_COLUMN(float, probP))

using MtdSoA = MtdSoALayout<>;
using MtdSoAView = MtdSoA::View;
using MtdSoAConstView = MtdSoA::ConstView;

#endif
