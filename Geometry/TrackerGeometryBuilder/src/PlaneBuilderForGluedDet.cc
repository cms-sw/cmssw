#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderForGluedDet.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"

#include <algorithm>

// Warning, remember to assign this pointer to a ReferenceCountingPointer!
PlaneBuilderForGluedDet::ResultType
PlaneBuilderForGluedDet::plane( const std::vector<const GeomDetUnit*>& dets ) const
{
  // find mean position
  typedef Surface::PositionType::BasicVectorType Vector;
  Vector posSum( 0, 0, 0 );
  for(auto det : dets)
  {
    posSum += (*det).surface().position().basicVector();
  }
  Surface::PositionType meanPos( posSum / float( dets.size()));
  
  Surface::RotationType rotation =  dets.front()->surface().rotation();
  //  Surface::RotationType rotation = computeRotation( dets, meanPos);
  Plane tmpPlane = Plane( meanPos, rotation);

  // Take the medium properties from the first DetUnit 
  const MediumProperties &  mp = dets.front()->surface().mediumProperties();
  MediumProperties newmp( mp.radLen() * 2.0, mp.xi() * 2.0 );

  std::pair<RectangularPlaneBounds*, GlobalVector> bo = computeRectBounds( dets, tmpPlane);
  return new Plane( meanPos+bo.second, rotation, newmp, bo.first);
}


std::pair<RectangularPlaneBounds*, GlobalVector>
PlaneBuilderForGluedDet::computeRectBounds( const std::vector<const GeomDetUnit*>& dets, const Plane& plane ) const
{
  // go over all corners and compute maximum deviations from mean pos.
  std::vector<GlobalPoint> corners;
  for(auto det : dets)
  {
    const Plane& bplane = dynamic_cast<const Plane&>(det->surface());
    std::vector<GlobalPoint> dc = BoundingBox().corners( bplane );
    corners.insert( corners.end(), dc.begin(), dc.end());
  }
  
  float xmin(0), xmax(0), ymin(0), ymax(0), zmin(0), zmax(0);
  for( std::vector<GlobalPoint>::const_iterator i = corners.begin(), cend = corners.end();
       i != cend; ++i )
  {
    LocalPoint p = plane.toLocal( *i );
    if( p.x() < xmin ) xmin = p.x();
    if( p.x() > xmax ) xmax = p.x();
    if( p.y() < ymin ) ymin = p.y();
    if( p.y() > ymax ) ymax = p.y();
    if( p.z() < zmin ) zmin = p.z();
    if( p.z() > zmax ) zmax = p.z();
  }

  LocalVector localOffset(( xmin + xmax ) / 2., ( ymin + ymax ) / 2., ( zmin + zmax ) / 2. );
  GlobalVector offset( plane.toGlobal( localOffset ));

  std::pair<RectangularPlaneBounds*, GlobalVector> result(new RectangularPlaneBounds(( xmax - xmin ) / 2, ( ymax - ymin ) / 2, ( zmax - zmin ) / 2 ), offset );

  return result;
}

Surface::RotationType
PlaneBuilderForGluedDet::computeRotation( const std::vector<GeomDetUnit const*>& dets, const Surface::PositionType& meanPos) const
{
  // choose first mono out-pointing rotation
  // the rotations of GluedDets coincide with the mono part
  // Simply take the x,y of the first Det if z points out,
  // or -x, y if it doesn't
  const BoundPlane& plane = dynamic_cast<const BoundPlane&>( dets.front()->surface());
  //GlobalVector n = plane.normalVector();

  GlobalVector xAxis;
  GlobalVector yAxis;
  GlobalVector planeYAxis = plane.toGlobal( LocalVector( 0, 1, 0 ));
  if( planeYAxis.z() < 0 )
    yAxis = -planeYAxis;
  else
    yAxis =  planeYAxis;

  GlobalVector planeXAxis = plane.toGlobal( LocalVector( 1, 0, 0 ));
  GlobalVector n = planeXAxis.cross( planeYAxis );

  if( n.x() * meanPos.x() + n.y() * meanPos.y() > 0 )
  {
    xAxis = planeXAxis;
  }
  else
  {
    xAxis = -planeXAxis;
  }

  return Surface::RotationType( xAxis, yAxis );
}
