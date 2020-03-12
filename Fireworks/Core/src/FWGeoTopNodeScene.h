#ifndef Fireworks_Core_FWGeoTopNodeScene_h
#define Fireworks_Core_FWGeoTopNodeScene_h

#include "TGLScenePad.h"

class FWGeoTopNode;
class TGLViewer;

class FWGeoTopNodeGLScene : public TGLScenePad {
private:
  FWGeoTopNodeGLScene(const FWGeoTopNodeGLScene&) = delete;             // Not implemented
  FWGeoTopNodeGLScene& operator=(const FWGeoTopNodeGLScene&) = delete;  // Not implemented
protected:
public:
  // UInt_t fNextCompositeID;
  FWGeoTopNode* m_eveTopNode;

  FWGeoTopNodeGLScene(TVirtualPad* pad);
  ~FWGeoTopNodeGLScene() override {}

  void SetPad(TVirtualPad* p) { fPad = p; }

  void GeoPopupMenu(Int_t gx, Int_t gy, TGLViewer*);

  using TGLScenePad::ResolveSelectRecord;
  Bool_t ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx) override;

  bool OpenCompositeWithPhyID(UInt_t phyID, const TBuffer3D& buffer);

  // virtual DestroyPhysicals() ... call m_eveTopNode->ClearSelectionHighlight;
  // There: if selected => gEve->GetSelection()->Remove(this) or sth
  //        if highlighted .... "" .....

  using TGLScenePad::DestroyPhysicals;
  Int_t DestroyPhysicals() override;
  using TGLScenePad::DestroyPhysical;
  virtual Bool_t DestroyPhysical(Int_t);

  using TGLScenePad::AddObject;
  Int_t AddObject(const TBuffer3D& buffer, Bool_t* addChildren = nullptr) override;
};

//==============================================================================
//==============================================================================
//==============================================================================
#if ROOT_VERSION_CODE < ROOT_VERSION(5, 32, 0)

#include "TEveScene.h"
class FWGeoTopNodeEveScene : public TEveScene {
public:
  FWGeoTopNodeEveScene(FWGeoTopNodeGLScene* gl_scene, const char* n = "TEveScene", const char* t = "");

  ~FWGeoTopNodeEveScene() {}
};
#endif

#endif
