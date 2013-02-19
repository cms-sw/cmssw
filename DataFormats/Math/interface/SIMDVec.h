#ifndef DataFormat_Math_SIMDVec_H
#define DataFormat_Math_SIMDVec_H


#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)
#define USE_EXTVECT
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) 
#define USE_SSEVECT
#endif




#if defined(USE_EXTVECT)  
#include "DataFormats/Math/interface/ExtVec.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/Math/interface/SSEVec.h"
#include "DataFormats/Math/interface/SSERot.h"
#endif

#endif //
