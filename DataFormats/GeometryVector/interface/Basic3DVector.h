#ifndef GeometryVector_Basic3DVector_h
#define GeometryVector_Basic3DVector_h

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)
#define USE_EXTVECT
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) 
#define USE_SSEVECT
#endif




#if ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometryVector/interface/oldBasic3DVector.h"
#elif defined(USE_EXTVECT)       
#include "DataFormats/GeometryVector/interface/extBasic3DVector.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometryVector/interface/sseBasic3DVector.h"
#endif

#endif // GeometryVector_Basic3DVector_h

