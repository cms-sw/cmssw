#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

using namespace align;

AlignableSurface::AlignableSurface(const BoundPlane& surface):
  GloballyPositioned<Scalar>( surface.position(), surface.rotation() ),
  theWidth( surface.bounds().width() ),
  theLength( surface.bounds().length() )
{
}

AlignableSurface::AlignableSurface(const PositionType& pos,
				   const RotationType& rot):
  GloballyPositioned<Scalar>(pos, rot),
  theWidth( Scalar() ),
  theLength( Scalar() )
{
}

GlobalPoints AlignableSurface::toGlobal(const LocalPoints& localPoints) const
{
  GlobalPoints globalPoints;

  unsigned int nPoint = localPoints.size();

  globalPoints.reserve(nPoint);

  for (unsigned int j = 0; j < nPoint; ++j)
  {
    globalPoints.push_back( toGlobal(localPoints[j]) );
  }

  return globalPoints;
}

RotationType AlignableSurface::toGlobal(const RotationType& rot) const
{
  return rotation().multiplyInverse( rot * rotation() );
}

EulerAngles AlignableSurface::toGlobal(const EulerAngles& angles) const
{
  return toAngles( toGlobal( toMatrix(angles) ) );
}

RotationType AlignableSurface::toLocal(const RotationType& rot) const
{
  return rotation() *  rot * rotation().transposed();
}

EulerAngles AlignableSurface::toLocal(const EulerAngles& angles) const
{
  return toAngles( toLocal( toMatrix(angles) ) );
}
