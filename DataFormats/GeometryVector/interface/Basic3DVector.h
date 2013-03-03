#ifndef GeometryVector_Basic3DVector_h
#define GeometryVector_Basic3DVector_h

#include "DataFormats/Math/interface/SIMDVec.h"


#if ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometryVector/interface/oldBasic3DVector.h"
#elif defined(USE_EXTVECT)       
#include "DataFormats/GeometryVector/interface/extBasic3DVector.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometryVector/interface/sseBasic3DVector.h"
#endif

#endif // GeometryVector_Basic3DVector_h

