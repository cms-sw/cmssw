#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

AlignableSurface::AlignableSurface(const PositionType& pos,
				   const RotationType& rot):
  GloballyPositioned<float>(pos, rot),
  theWidth(0.),
  theLength(0.)
{
}

std::vector<AlignableSurface::GlobalPoint> AlignableSurface::toGlobal(const std::vector<LocalPoint>& localPoints) const
{
  std::vector<GlobalPoint> globalPoints;

  globalPoints.reserve( localPoints.size() );

  globalPoints.push_back( toGlobal(localPoints[0]) );
  globalPoints.push_back( toGlobal(localPoints[1]) );
  globalPoints.push_back( toGlobal(localPoints[2]) );
  globalPoints.push_back( toGlobal(localPoints[3]) );
  globalPoints.push_back( toGlobal(localPoints[4]) );
  globalPoints.push_back( toGlobal(localPoints[5]) );
  globalPoints.push_back( toGlobal(localPoints[6]) );
  globalPoints.push_back( toGlobal(localPoints[7]) );
  globalPoints.push_back( toGlobal(localPoints[8]) );

  return globalPoints;
}

TkRotation<float> AlignableSurface::toLocal(const TkRotation<float>& rot) const
{
  return rotation() *  rot * rotation().transposed();
}
