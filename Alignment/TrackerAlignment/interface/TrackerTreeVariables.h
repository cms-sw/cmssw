#ifndef TrackerTreeVariables_h
#define TrackerTreeVariables_h

// For ROOT types with '_t':
#include <Rtypes.h>

// container to hold all static module parameters, determined with ideal geometry
struct TrackerTreeVariables {
  TrackerTreeVariables() { this->clear(); }

  void clear() {
    rawId = subdetId = layer = side = half = rod = ring = petal = blade = panel = outerInner = module = nStrips = 0;
    isDoubleSide = isRPhi = isStereo = false;
    uDirection = vDirection = wDirection = 0;
    posR = posPhi = posEta = posX = posY = posZ = -999.F;
  }

  UInt_t rawId, subdetId, layer, side, half, rod, ring, petal, blade, panel, outerInner, module, nStrips;
  Bool_t isDoubleSide, isRPhi, isStereo;
  Int_t uDirection, vDirection, wDirection;
  Float_t posR, posPhi, posEta, posX, posY, posZ;
};

#endif
