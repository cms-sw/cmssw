#ifndef Geom_TkRotation_H
#define Geom_TkRotation_H


#include "DataFormats/Math/interface/SIMDVec.h"

#if defined(USE_EXTVECT)  
#include "DataFormats/GeometrySurface/interface/extTkRotation.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/GeometrySurface/interface/sseTkRotation.h"
#else
#include "DataFormats/GeometrySurface/interface/oldTkRotation.h"
#endif

#endif // Geom_TkRotation_H

