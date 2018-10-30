#include "Geometry/CommonDetUnit/interface/MTDGeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

bool MTDGeomDet::setAlignmentPositionError (const AlignmentPositionError& ape) 
{
  if (!theAlignmentPositionError) {
    if (ape.valid()) theAlignmentPositionError = new AlignmentPositionError(ape);
  } 
  else *theAlignmentPositionError = ape;

  const GlobalErrorExtended& apeError = ape.globalError();
  GlobalError translatApe(apeError.cxx(),apeError.cyx(),apeError.cyy(),apeError.czx(),apeError.czy(),apeError.czz());

  //check only translat part is valid 
  theLocalAlignmentError = ape.valid() ?
    ErrorFrameTransformer().transform( translatApe,surface()) :
    InvalidError();
  return ape.valid();
}

