#ifndef Geom_TkRotation_H
#define Geom_TkRotation_H

#include "DataFormats/Math/interface/SIMDVec.h"

#if defined(USE_EXTVECT)
#include "private/extTkRotation.h"
#elif defined(USE_SSEVECT)
#include "private/sseTkRotation.h"
#else
#include "private/oldTkRotation.h"
#endif

#endif  // Geom_TkRotation_H
