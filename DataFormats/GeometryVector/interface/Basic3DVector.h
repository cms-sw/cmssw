#ifndef GeometryVector_Basic3DVector_h
#define GeometryVector_Basic3DVector_h

#include "DataFormats/Math/interface/SIMDVec.h"

#if defined(USE_EXTVECT)
#include "private/extBasic3DVector.h"
#elif defined(USE_SSEVECT)
#include "private/sseBasic3DVector.h"
#else
#include "private/oldBasic3DVector.h"
#endif

#endif  // GeometryVector_Basic3DVector_h
