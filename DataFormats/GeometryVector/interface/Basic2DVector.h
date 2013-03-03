#ifndef GeometryVector_Basic2DVector_h
#define GeometryVector_Basic2DVector_h
#include "DataFormats/Math/interface/SIMDVec.h"

#if ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometryVector/interface/oldBasic2DVector.h"
#elif defined(USE_EXTVECT)       
#include "DataFormats/GeometryVector/interface/extBasic2DVector.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometryVector/interface/sseBasic2DVector.h"
#endif

#endif // GeometryVector_Basic2DVector_h

