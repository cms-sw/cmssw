#ifndef GeometryVector_Basic2DVector_h
#define GeometryVector_Basic2DVector_h

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)
#define USE_SSEVECT
#endif

#if defined(USE_SSEVECT) && ! ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometryVector/interface/newBasic2DVector.h"
#else
#include "DataFormats/GeometryVector/interface/oldBasic2DVector.h"
#endif

#endif // GeometryVector_Basic2DVector_h

