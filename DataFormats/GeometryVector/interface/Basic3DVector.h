#ifndef GeometryVector_Basic3DVector_h
#define GeometryVector_Basic3DVector_h

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) || defined(__clang__)
#define USE_SSEVECT
#endif

#if defined(USE_SSEVECT) && ! ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometryVector/interface/newBasic3DVector.h"
#else
#include "DataFormats/GeometryVector/interface/oldBasic3DVector.h"
#endif

#endif // GeometryVector_Basic3DVector_h


