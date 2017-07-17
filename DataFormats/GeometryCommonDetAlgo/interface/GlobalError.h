#ifndef _TRACKER_GLOBAL_ERROR_H_
#define _TRACKER_GLOBAL_ERROR_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalErrorBase.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalErrorBaseExtended.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorMatrixTag.h"

  /**
   * typedef to a  GlobalErrorBase object defined as a 4*4 covariance matrix
   * acts like a 3*3 matrix to preserve backwards compatibility
   */

typedef GlobalErrorBase<double,ErrorMatrixTag> GlobalError;
typedef GlobalErrorBaseExtended<double,ErrorMatrixTag> GlobalErrorExtended;

#endif
