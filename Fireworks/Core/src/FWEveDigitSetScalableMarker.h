#ifndef Fireworks_Core_FWMarkerDigitSetGL_h
#define Fireworks_Core_FWMarkerDigitSetGL_h

#include "TEveQuadSet.h"
#include "TEveQuadSetGL.h"
#include "TGLRnrCtx.h"
#include "TAttMarker.h"

class FWEveDigitSetScalableMarker : public TEveQuadSet, public TAttMarker {
public:
  FWEveDigitSetScalableMarker() {}
  ~FWEveDigitSetScalableMarker() override {}

  ClassDefOverride(FWEveDigitSetScalableMarker, 0);
};

//--------------------------------------------
class FWEveDigitSetScalableMarkerGL : public TEveQuadSetGL {
public:
  FWEveDigitSetScalableMarkerGL() {}
  ~FWEveDigitSetScalableMarkerGL() override {}

  void DirectDraw(TGLRnrCtx& rnrCtx) const override;

  ClassDefOverride(FWEveDigitSetScalableMarkerGL, 0);
};

#endif
