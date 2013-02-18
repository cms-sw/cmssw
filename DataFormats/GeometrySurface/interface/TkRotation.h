#ifndef Geom_TkRotation_H
#define Geom_TkRotation_H

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)
#define USE_EXTVECT
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) 
#define USE_SSEVECT
#endif




#if ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometrySurface/interface/oldTkRotation.h"
#elif defined(USE_EXTVECT)  
#include "DataFormats/GeometrySurface/interface/extTkRotation.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometrySurface/interface/sseTkRotation.h"
#endif

#endif // Geom_TkRotation_H

