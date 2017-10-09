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
   virtual ~FWGeoTopNodeGL() {}

   virtual void   SetBBox();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   DirectDraw(TGLRnrCtx& rnrCtx) const;

   // virtual void   DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl=-1) const;

   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual Bool_t AlwaysSecondarySelect()   const { return kTRUE; }
   virtual void   ProcessSelection(TGLRnrCtx& rnrCtx, TGLSelectRecord& rec);

   ClassDef(FWGeoTopNodeGL, 0); // GL renderer class for FWGeoTopNodeGL.

};
#endif
