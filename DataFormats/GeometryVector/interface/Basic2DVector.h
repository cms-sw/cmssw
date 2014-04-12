#ifndef GeometryVector_Basic2DVector_h
#define GeometryVector_Basic2DVector_h
#include "DataFormats/Math/interface/SIMDVec.h"

#if defined(USE_EXTVECT)       
#include "DataFormats/GeometryVector/interface/extBasic2DVector.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometryVector/interface/sseBasic2DVector.h"
#else
#include "DataFormats/GeometryVector/interface/oldBasic2DVector.h"
#endif

#endif // GeometryVector_Basic2DVector_h

