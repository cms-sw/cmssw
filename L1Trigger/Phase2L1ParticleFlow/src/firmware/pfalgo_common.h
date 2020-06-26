#ifndef L1Trigger_Phase2L1ParticleFlow_FIRMWARE_PFALGO_COMMON_H
#define L1Trigger_Phase2L1ParticleFlow_FIRMWARE_PFALGO_COMMON_H

#include "data.h"

inline int dr2_int(etaphi_t eta1, etaphi_t phi1, etaphi_t eta2, etaphi_t phi2) {
  ap_int<etaphi_t::width + 1> deta = (eta1 - eta2);
  ap_int<etaphi_t::width + 1> dphi = (phi1 - phi2);
  return deta * deta + dphi * dphi;
}

#ifndef CMSSW_GIT_HASH
#define PFALGO_DR2MAX_TK_MU 2101
#endif

#endif
