#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

bool TrackerGeomDet::setAlignmentPositionError (const AlignmentPositionError& ape) 
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

