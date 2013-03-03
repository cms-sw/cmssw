#ifndef Geom_TkRotation_H
#define Geom_TkRotation_H


#include "DataFormats/Math/interface/SIMDVec.h"



#if ( defined(__REFLEX__) || defined(__CINT__) )
#include "DataFormats/GeometrySurface/interface/oldTkRotation.h"
#elif defined(USE_EXTVECT)  
#include "DataFormats/GeometrySurface/interface/extTkRotation.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometrySurface/interface/sseTkRotation.h"
#endif

#endif // Geom_TkRotation_H

