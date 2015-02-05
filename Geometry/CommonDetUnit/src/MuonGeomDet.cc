#include "Geometry/CommonDetUnit/interface/MuonGeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

bool MuonGeomDet::setAlignmentPositionError (const AlignmentPositionError& ape) 
{
  //this is a placeholder, global to local conversion is done in the MuonTransientTrackingRecHit at the moment
  return true;
}
