/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"

//----------------------------------------------------------------------------------------------------

bool operator< (const TotemRPUVPattern &l, const TotemRPUVPattern &r)
{
  if (l.projection < r.projection)
    return true;
  if (l.projection > r.projection)
    return false;

  if (l.a < r.a)
    return true;
  if (l.a > r.a)
    return false;

  if (l.b < r.b)
    return true;
  if (l.b > r.b)
    return false;

  if (l.w < r.w)
    return true;
  if (l.w > r.w)
    return false;

  if (l.fittable < r.fittable)
    return true;
  if (l.fittable > r.fittable)
    return false;

  return false;
}
