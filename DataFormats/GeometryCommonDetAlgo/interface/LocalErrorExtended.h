#ifndef DataFormats_GeometryCommonDetAlgo_LocalErrorExtended_h
#define DataFormats_GeometryCommonDetAlgo_LocalErrorExtended_h

#include "DataFormats/GeometryCommonDetAlgo/interface/LocalErrorBaseExtended.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorMatrixTag.h"

/**
   * typedef to a  GlobalErrorBase object defined as a 3*3 covariance matrix
   */

typedef LocalErrorBaseExtended<double, ErrorMatrixTag> LocalErrorExtended;

#endif
