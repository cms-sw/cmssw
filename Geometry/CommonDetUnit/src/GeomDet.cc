#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/ModifiedSurfaceGenerator.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

GeomDet::GeomDet( Plane* plane):
  thePlane(plane), theAlignmentPositionError(0), theLocalAlignmentError(InvalidError()), m_index(-1) {}

GeomDet::GeomDet( const ReferenceCountingPointer<Plane>& plane) :
  thePlane(plane), theAlignmentPositionError(0), theLocalAlignmentError(InvalidError()), m_index(-1) {}

GeomDet::~GeomDet() {delete theAlignmentPositionError;}

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
  thePlane = ModifiedSurfaceGenerator<Plane>(thePlane).atNewPosition(position,
									  rotation);
}

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
bool GeomDet::setAlignmentPositionError (const AlignmentPositionError& ape) 
{
  if (!theAlignmentPositionError) {
    if (ape.valid()) theAlignmentPositionError = new AlignmentPositionError(ape);
  } 
  else *theAlignmentPositionError = ape;

  theLocalAlignmentError = ape.valid() ?
    ErrorFrameTransformer().transform( ape.globalError(),
                                       surface()
				       ) :
    InvalidError();
  return ape.valid();
}
