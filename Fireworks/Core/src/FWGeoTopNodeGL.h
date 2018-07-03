#ifndef Fireworks_Core_FWGeoTopNodeGL_h
#define Fireworks_Core_FWGeoTopNodeGL_h

#include "TGLObject.h"
class FWGeoTopNode;

class FWGeoTopNodeGL : public TGLObject
{
protected:
   FWGeoTopNode     *fM;

public:
   FWGeoTopNodeGL();
   ~FWGeoTopNodeGL() override {}

   void   SetBBox() override;

   Bool_t SetModel(TObject* obj, const Option_t* opt=nullptr) override;
   void   DirectDraw(TGLRnrCtx& rnrCtx) const override;

   // virtual void   DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl=-1) const;

   Bool_t SupportsSecondarySelect() const override { return kTRUE; }
   Bool_t AlwaysSecondarySelect()   const override { return kTRUE; }
   void   ProcessSelection(TGLRnrCtx& rnrCtx, TGLSelectRecord& rec) override;

   ClassDefOverride(FWGeoTopNodeGL, 0); // GL renderer class for FWGeoTopNodeGL.

};
#endif
