#include "Fireworks/Core/src/FWGeoTopNodeScene.h"
#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "TGLSelectRecord.h"
#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "TGeoVolume.h"
#include "TBuffer3D.h"


//______________________________________________________________________________
FWGeoTopNodeGLScene::FWGeoTopNodeGLScene(TVirtualPad* pad) :
   TGLScenePad(pad),
   // fNextCompositeID(0),
   fTopNodeJebo(0)
{
   // Constructor.
   // fInternalPIDs = false;
   fTitle="JeboScene";
}

//______________________________________________________________________________
Bool_t FWGeoTopNodeGLScene::OpenCompositeWithPhyID(UInt_t phyID, const TBuffer3D& buffer)
{
   // Open new composite container.
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture.

   if (fComposite) {
      Error("FWGeoTopNodeGLScene::OpenComposite", "composite already open");
      return kFALSE;
   }

   UInt_t extraSections = TGLScenePad::AddObject(phyID, buffer, 0);
   if (extraSections != TBuffer3D::kNone) {
      Error("FWGeoTopNodeGLScene::OpenComposite", "expected top level composite to not require extra buffer sections");
   }

   // If composite was created it is of interest - we want the rest of the
   // child components
   if (fComposite) {
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Int_t FWGeoTopNodeGLScene::AddObject(const TBuffer3D& buffer, Bool_t* addChildren)
{
   if (fComposite)
   {
      // TGeoSubstraction, TGeoUnion, ... phyID ignored in this case
      int ns = TGLScenePad::AddObject(1, buffer, addChildren);
      return ns;
   }
   else
   {  
      fwLog(fwlog::kError)<< "FWGeoTopNodeGLScene::AddObject() should not be called if fNextCompositeID \n";
      return TGLScenePad::AddObject(buffer, addChildren);
   }
}

//______________________________________________________________________________
Bool_t FWGeoTopNodeGLScene::ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx)
{
   // Process selection record rec.
   // 'curIdx' is the item position where the scene should start
   // its processing.
   // Return TRUE if an object has been identified or FALSE otherwise.
   // The scene-info member of the record is already set by the caller.  

   if (curIdx >= rec.GetN())
      return kFALSE;

   TGLPhysicalShape* pshp = FindPhysical(rec.GetItem(curIdx));

   /*
   printf("FWGeoTopNodeGLScene::ResolveSelectRecord pshp=%p, lshp=%p, obj=%p, shpcls=%s\n",
          (void*)pshp,(void*) pshp->GetLogical(),(void*) pshp->GetLogical()->GetExternal(),
          ((TGeoVolume*)pshp->GetLogical()->GetExternal())->GetShape()->ClassName());
   */
   if (pshp)
   {
      rec.SetTransparent(pshp->IsTransparent());
      rec.SetPhysShape(pshp);

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,32,0)
      rec.SetLogShape(FindLogical(fTopNodeJebo));
#endif
      rec.SetObject(fTopNodeJebo);
      rec.SetSpecific(0);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void FWGeoTopNodeGLScene::GeoPopupMenu(Int_t gx, Int_t gy)
{fTopNodeJebo->popupMenu(gx, gy);
}

//==============================================================================
//==============================================================================
//==============================================================================
#if ROOT_VERSION_CODE < ROOT_VERSION(5,32,0)

#include "TEvePad.h"
FWGeoTopNodeEveScene::FWGeoTopNodeEveScene(FWGeoTopNodeGLScene* gl_scene, const char* n, const char* t)
{
   // Constructor.

   delete fGLScene;

   gl_scene->SetPad(fPad);
   fGLScene = gl_scene;

   fGLScene->SetName(n);
   fGLScene->SetAutoDestruct(kFALSE);
   fGLScene->SetSmartRefresh(kTRUE);
}
#endif
