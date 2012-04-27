#ifndef Fireworks_Core_FWGeoTopNodeScene_h
#define Fireworks_Core_FWGeoTopNodeScene_h


#include "TGLScenePad.h"

class FWGeoTopNode;
class FWGeoTopNode;

class FWGeoTopNodeGLScene : public TGLScenePad
{
private:
   FWGeoTopNodeGLScene(const FWGeoTopNodeGLScene&);            // Not implemented
   FWGeoTopNodeGLScene& operator=(const FWGeoTopNodeGLScene&); // Not implemented
protected:

public:
   // UInt_t fNextCompositeID;
   FWGeoTopNode *fTopNodeJebo;

   FWGeoTopNodeGLScene(TVirtualPad* pad);
   virtual ~FWGeoTopNodeGLScene() {}

   void SetPad(TVirtualPad* p) { fPad = p; }

   void  GeoPopupMenu(Int_t gx, Int_t gy);

   virtual Bool_t ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx);

   bool OpenCompositeWithPhyID(UInt_t phyID, const TBuffer3D& buffer);

   virtual Int_t  AddObject(const TBuffer3D& buffer, Bool_t* addChildren = 0);
};
#endif
