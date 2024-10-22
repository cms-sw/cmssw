#ifndef L1Trigger_Phase2L1ParticleFlow_LINPUPPI_BITS_H
#define L1Trigger_Phase2L1ParticleFlow_LINPUPPI_BITS_H

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"

namespace linpuppi {
  typedef ap_ufixed<12, 6, AP_TRN, AP_SAT> sumTerm_t;
  typedef ap_ufixed<16, 0, AP_RND, AP_SAT> dr2inv_t;
  typedef ap_fixed<12, 7, AP_TRN, AP_SAT> x2_t;
  typedef ap_ufixed<7, 2, AP_RND, AP_WRAP> alphaSlope_t;
  typedef ap_fixed<12, 8, AP_RND, AP_WRAP> alpha_t;
  typedef ap_ufixed<6, 0, AP_TRN, AP_WRAP> ptSlope_t;

  constexpr float DR2_LSB = l1ct::Scales::ETAPHI_LSB * l1ct::Scales::ETAPHI_LSB;
  constexpr float PT2DR2_LSB = l1ct::Scales::INTPT_LSB * l1ct::Scales::INTPT_LSB / DR2_LSB;
  constexpr int SUM_BITSHIFT = sumTerm_t::width - sumTerm_t::iwidth;
}  // namespace linpuppi

#endif
