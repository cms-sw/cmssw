#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhiThetaPair.h"

L1Phase2MuDTExtPhiThetaPair::L1Phase2MuDTExtPhiThetaPair(const L1Phase2MuDTExtPhDigi& phi,
                                                         const L1Phase2MuDTExtThDigi& theta,
                                                         int quality)
    : phi_(phi), theta_(theta), quality_(quality) {}
