#ifndef TRK_SELECTION_H
#define TRK_SELECTION_H

#include <utility>
#include "DQM/PhysicsHWW/interface/HWW.h"

namespace HWWFunctions {

  std::pair<double , double> trks_d0_pv      (HWW&, int itrk, int ipv);
  std::pair<double , double> trks_dz_pv      (HWW&, int itrk, int ipv);
  std::pair<double , double> gsftrks_dz_pv   (HWW&, int itrk, int ipv);
  std::pair<double , double> gsftrks_d0_pv   (HWW&, int itrk, int ipv);

}
#endif
