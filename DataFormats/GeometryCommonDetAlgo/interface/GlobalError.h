#ifndef _TRACKER_GLOBAL_ERROR_H_
#define _TRACKER_GLOBAL_ERROR_H_

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalErrorBase4D.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalErrorBase.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorMatrixTag.h"

  /**
   * typedef to a  GlobalErrorBase object defined as a 3*3 covariance matrix
   * typedef to a  GlobalErrorBase4D object defined as a 4*4 covariance matrix
   */

typedef GlobalErrorBase<double,ErrorMatrixTag> GlobalError;
typedef GlobalErrorBase4D<double,ErrorMatrixTag> GlobalError4D;

#endif
