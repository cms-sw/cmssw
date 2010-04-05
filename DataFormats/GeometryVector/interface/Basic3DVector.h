#ifndef GeometryVector_Basic3DVector_h
#define GeometryVector_Basic3DVector_h

#ifdef USE_SSE
#include "DataFormats/GeometryVector/interface/newBasic3DVector.h"
#include "DataFormats/GeometryVector/interface/Basic3DVectorFSSE.icc"
#else
#include "DataFormats/GeometryVector/interface/oldBasic3DVector.h"
#endif

#endif // GeometryVector_Basic3DVector_h


