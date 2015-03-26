#ifndef _TRACKER_LOCAL_ERROR_H_
#define _TRACKER_LOCAL_ERROR_H_

#include "DataFormats/GeometrySurface/interface/LocalErrorBaseExtended.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorMatrixTag.h"

  /**
   * typedef to a  GlobalErrorBase object defined as a 3*3 covariance matrix
   */

typedef LocalErrorBaseExtended<double,ErrorMatrixTag> LocalErrorExtended;

#endif
