#ifndef FIRMWARE_PFALGO_TYPES_H
#define FIRMWARE_PFALGO_TYPES_H

#ifdef CMSSW_GIT_HASH
#include "../dataformats/datatypes.h"
#else
#include "../../dataformats/datatypes.h"
#endif

namespace l1ct {

  typedef ap_ufixed<17, 17 - 4, AP_TRN, AP_SAT> ptscale_t;
  typedef ap_ufixed<9, 1> ptErrScale_t;
  typedef ap_fixed<10, 6> ptErrOffs_t;

}  // namespace l1ct

#endif
