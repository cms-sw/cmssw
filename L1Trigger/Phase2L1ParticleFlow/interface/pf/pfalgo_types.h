#ifndef FIRMWARE_PFALGO_TYPES_H
#define FIRMWARE_PFALGO_TYPES_H

#include "DataFormats/L1TParticleFlow/interface/datatypes.h"

namespace l1ct {

  typedef ap_ufixed<17, 17 - 4, AP_TRN, AP_SAT> ptscale_t;
  typedef ap_ufixed<9, 1> ptErrScale_t;
  typedef ap_fixed<10, 6> ptErrOffs_t;

}  // namespace l1ct

#endif
