/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/


#include <cmath>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSTimingLocalTrack.h"

//----------------------------------------------------------------------------------------------------

//====================================================================================================
// CTPPSTimingLocalTrack class implementation
//====================================================================================================

//--- constructors

CTPPSTimingLocalTrack::CTPPSTimingLocalTrack() :
  num_hits_(0),
  num_planes_(0),
  valid_(true),
  t_(0.),
  t_sigma_(0.) {}

CTPPSTimingLocalTrack::CTPPSTimingLocalTrack(
    const math::XYZPoint& pos0,
    const math::XYZPoint& pos0_sigma,
    float t,
    float t_sigma) :
  pos0_(pos0),
  pos0_sigma_(pos0_sigma),
  num_hits_(0),
  num_planes_(0),
  valid_(false),
  t_(t),
  t_sigma_(t_sigma) {}


//--- interface member functions

bool CTPPSTimingLocalTrack::containsHit(
    const CTPPSTimingRecHit& recHit,
    float tolerance,
    CheckDimension check
  ) const {

  float xTolerance = pos0_sigma_.x() + (0.5 * recHit.getXWidth()) + tolerance;
  float yTolerance = pos0_sigma_.y() + (0.5 * recHit.getYWidth()) + tolerance;
  float zTolerance = pos0_sigma_.z() + (0.5 * recHit.getZWidth()) + tolerance;

  float xDiff = fabs(pos0_.x() - recHit.getX());
  float yDiff = fabs(pos0_.y() - recHit.getY());
  float zDiff = fabs(pos0_.z() - recHit.getZ());

  if(check == CHECK_X)
    return xDiff < xTolerance;

  if(check == CHECK_Y)
    return yDiff < yTolerance;

  return (xDiff < xTolerance && yDiff < yTolerance && zDiff < zTolerance);
}


//====================================================================================================
// Other methods implementation
//====================================================================================================

bool operator<( const CTPPSTimingLocalTrack& lhs, const CTPPSTimingLocalTrack& rhs )
{
  // start to sort by temporal coordinate
  if ( lhs.getT() < rhs.getT() ) return true;
  if ( lhs.getT() > rhs.getT() ) return false;
  // then sort by x-position
  if ( lhs.getX0() < rhs.getX0() ) return true;
  if ( lhs.getX0() > rhs.getX0() ) return false;
  // ...and y-position
  if ( lhs.getY0() < rhs.getY0() ) return true;
  if ( lhs.getY0() > rhs.getY0() ) return false;
  // ...and z-position
  return ( lhs.getZ0() < rhs.getZ0() );
}
