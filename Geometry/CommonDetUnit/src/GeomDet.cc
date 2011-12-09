#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

GeomDet::GeomDet( BoundPlane* plane):
  thePlane(plane), theLocalAlignmentError(InvalidError()) {}

GeomDet::GeomDet( const ReferenceCountingPointer<BoundPlane>& plane) :
  thePlane(plane), theLocalAlignmentError(InvalidError()) {}

GeomDet::~GeomDet() {}

void GeomDet::move( const GlobalVector& displacement)
{
  //
  // Should recreate the surface like the set* methods ?
  //
  thePlane->move(displacement);
}

void GeomDet::rotate( const Surface::RotationType& rotation)
{
  //
  // Should recreate the surface like the set* methods ?
  //
  thePlane->rotate(rotation);
}

void GeomDet::setPosition( const Surface::PositionType& position, 
			   const Surface::RotationType& rotation)
{
  thePlane = ModifiedSurfaceGenerator<BoundPlane>(thePlane).atNewPosition(position,
									  rotation);
}

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
bool GeomDet::setAlignmentPositionError (const AlignmentPositionError& ape) 
{
  theLocalAlignmentError = ape.valid() ?
    ErrorFrameTransformer().transform( ape.globalError(),
                                       surface()
				       ) :
    InvalidError();
  return ape.valid();
}
