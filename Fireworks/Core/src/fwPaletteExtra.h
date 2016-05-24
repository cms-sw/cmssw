#ifndef Fireworks_Core_FWColorManagerExtra_h
#define Fireworks_Core_FWColorManagerExtra_h
#include "Fireworks/Core/interface/FWColorManager.h"

namespace fireworks {
void 
GetColorValuesForPaletteExtra(float(* iColors)[3], unsigned int iSize, FWColorManager::EPalette id, bool isBlack);
}
#endif

