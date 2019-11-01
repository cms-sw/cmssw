/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"

//----------------------------------------------------------------------------------------------------

bool operator<(const TotemRPUVPattern &l, const TotemRPUVPattern &r) {
  if (l.projection_ < r.projection_)
    return true;
  if (l.projection_ > r.projection_)
    return false;

  if (l.a_ < r.a_)
    return true;
  if (l.a_ > r.a_)
    return false;

  if (l.b_ < r.b_)
    return true;
  if (l.b_ > r.b_)
    return false;

  if (l.w_ < r.w_)
    return true;
  if (l.w_ > r.w_)
    return false;

  if (l.fittable_ < r.fittable_)
    return true;
  if (l.fittable_ > r.fittable_)
    return false;

  return false;
}
