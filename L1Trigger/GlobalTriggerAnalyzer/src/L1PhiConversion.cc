/**
 *
 * Description: math utilities.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1PhiConversion.h"

// convert phi from rad (-pi, pi] to deg (0, 360)
const double rad2deg(const double& phiRad) {
  if (phiRad < 0.) {
    return (phiRad * PiConversion) + 360.;
  } else {
    return (phiRad * PiConversion);
  }
}
