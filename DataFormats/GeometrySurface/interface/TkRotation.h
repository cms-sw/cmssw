#ifndef Geom_TkRotation_H
#define Geom_TkRotation_H

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) || defined(__clang__)
#define USE_SSEVECT
#endif

#if defined(USE_SSEVECT) && ! defined(__REFLEX__)
#include "DataFormats/GeometrySurface/interface/newTkRotation.h"
#else
#include "DataFormats/GeometrySurface/interface/oldTkRotation.h"
#endif

#endif // Geom_TkRotation_H

