#ifndef L1Trigger_GlobalTriggerAnalyzer_L1PhiConversion_h
#define L1Trigger_GlobalTriggerAnalyzer_L1PhiConversion_h

/**
 *
 * Description: math utilities.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <cmath>

static const double PiConversion = 180. / acos(-1.);

///  convert phi from rad (-pi, pi] to deg (0, 360)
const double rad2deg(const double&);

#endif
