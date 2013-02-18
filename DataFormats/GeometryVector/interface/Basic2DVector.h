#ifndef GeometryVector_Basic2DVector_h
#define GeometryVector_Basic2DVector_h

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)
#define USE_EXTVEC
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) 
#define USE_SSEVECT
#endif




#if ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometryVector/interface/oldBasic2DVector.h"
#elif defined(USE_EXTVECT)       
#include "DataFormats/GeometryVector/interface/extBasic2DVector.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometryVector/interface/sseBasic2DVector.h"
#endif

#endif // GeometryVector_Basic2DVector_h

