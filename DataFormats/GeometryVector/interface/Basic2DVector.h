#ifndef GeometryVector_Basic2DVector_h
#define GeometryVector_Basic2DVector_h
#include "DataFormats/Math/interface/SIMDVec.h"

#if defined(USE_EXTVECT)
#include "private/extBasic2DVector.h"
#elif defined(USE_SSEVECT)
#include "private/sseBasic2DVector.h"
#else
#include "private/oldBasic2DVector.h"
#endif

#endif  // GeometryVector_Basic2DVector_h
