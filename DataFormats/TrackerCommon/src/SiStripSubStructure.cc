// -*- C++ -*-
//
// Package:     SiStripDetId
// Class  :     SiStripSubStructure
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  dkcira
//         Created:  Wed Jan 25 07:19:38 CET 2006
//
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include <algorithm>

#include "DataFormats/TrackerCommon/interface/SiStripSubStructure.h"

void SiStripSubStructure::getTIBDetectors(const std::vector<uint32_t> &inputDetRawIds,
                                          std::vector<uint32_t> &tibDetRawIds,
                                          const TrackerTopology *tTopo,
                                          uint32_t rq_layer,
                                          uint32_t rq_bkw_frw,
                                          uint32_t rq_int_ext,
                                          uint32_t rq_string) {
  std::copy_if(std::begin(inputDetRawIds),
               std::end(inputDetRawIds),
               std::back_inserter(tibDetRawIds),
               [tTopo, rq_layer, rq_bkw_frw, rq_int_ext, rq_string](DetId det) {
                 return ((StripSubdetector::TIB == det.subdetId())
                         // check if TIB is from the ones requested
                         // take everything if default value is 0
                         && ((rq_layer == 0) || (rq_layer == tTopo->tibLayer(det))) &&
                         ((rq_bkw_frw == 0) || (rq_bkw_frw == tTopo->tibSide(det))) &&
                         ((rq_int_ext == 0) || (rq_int_ext == tTopo->tibOrder(det))) &&
                         ((rq_string == 0) || (rq_string == tTopo->tibString(det))));
               });
}

void SiStripSubStructure::getTIDDetectors(const std::vector<uint32_t> &inputDetRawIds,
                                          std::vector<uint32_t> &tidDetRawIds,
                                          const TrackerTopology *tTopo,
                                          uint32_t rq_side,
                                          uint32_t rq_wheel,
                                          uint32_t rq_ring,
                                          uint32_t rq_ster) {
  std::copy_if(std::begin(inputDetRawIds),
               std::end(inputDetRawIds),
               std::back_inserter(tidDetRawIds),
               [tTopo, rq_side, rq_wheel, rq_ring, rq_ster](DetId det) {
                 return ((StripSubdetector::TID == det.subdetId())
                         // check if TID is from the ones requested
                         // take everything if default value is 0
                         && ((rq_side == 0) || (rq_side == tTopo->tidSide(det))) &&
                         ((rq_wheel == 0) || (rq_wheel == tTopo->tidWheel(det))) &&
                         ((rq_ring == 0) || (rq_ring == tTopo->tidRing(det))) &&
                         ((rq_ster == 0) || (rq_ster == tTopo->tidStereo(det))));
               });
}

void SiStripSubStructure::getTOBDetectors(const std::vector<uint32_t> &inputDetRawIds,
                                          std::vector<uint32_t> &tobDetRawIds,
                                          const TrackerTopology *tTopo,
                                          uint32_t rq_layer,
                                          uint32_t rq_bkw_frw,
                                          uint32_t rq_rod) {
  std::copy_if(std::begin(inputDetRawIds),
               std::end(inputDetRawIds),
               std::back_inserter(tobDetRawIds),
               [tTopo, rq_layer, rq_bkw_frw, rq_rod](DetId det) {
                 return ((StripSubdetector::TOB == det.subdetId())
                         // check if TOB is from the ones requested
                         // take everything if default value is 0
                         && ((rq_layer == 0) || (rq_layer == tTopo->tobLayer(det))) &&
                         ((rq_bkw_frw == 0) || (rq_bkw_frw == tTopo->tobSide(det))) &&
                         ((rq_rod == 0) || (rq_rod == tTopo->tobRod(det))));
               });
}

void SiStripSubStructure::getTECDetectors(const std::vector<uint32_t> &inputDetRawIds,
                                          std::vector<uint32_t> &tecDetRawIds,
                                          const TrackerTopology *tTopo,
                                          uint32_t rq_side,
                                          uint32_t rq_wheel,
                                          uint32_t rq_petal_bkw_frw,
                                          uint32_t rq_petal,
                                          uint32_t rq_ring,
                                          uint32_t rq_ster) {
  std::copy_if(std::begin(inputDetRawIds),
               std::end(inputDetRawIds),
               std::back_inserter(tecDetRawIds),
               [tTopo, rq_side, rq_wheel, rq_petal_bkw_frw, rq_petal, rq_ring, rq_ster](DetId det) {
                 return ((StripSubdetector::TEC == det.subdetId())
                         // check if TEC is from the ones requested
                         // take everything if default value is 0
                         && ((rq_side == 0) || (rq_side == tTopo->tecSide(det))) &&
                         ((rq_wheel == 0) || (rq_wheel == tTopo->tecWheel(det))) &&
                         ((rq_petal_bkw_frw == 0) || (rq_petal_bkw_frw - 1 == tTopo->tecOrder(det))) &&
                         ((rq_petal == 0) || (rq_petal == tTopo->tecPetalNumber(det))) &&
                         ((rq_ring == 0) || (rq_ring == tTopo->tecRing(det))) &&
                         ((rq_ster == 0) || (rq_ster == tTopo->tecStereo(det))));
               });
}
