#ifndef Fireworks_Core_FWGeoTopNode_h
#define Fireworks_Core_FWGeoTopNode_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeoTopNode
// 
/**\class FWGeoTopNode FWGeoTopNode.h Fireworks/Core/interface/FWGeoTopNode.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Matevz Tadel, Alja Mrak Tadel
//         Created:  Thu Jun 23 01:25:00 CEST 2011
//

#ifndef __CINT__
#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
#endif
#include "TEveElement.h"
#include "TAttBBox.h"
#include "TGLUtil.h"
#include  <set>

class TGeoHMatrix;
class TGLPhysicalShape;
class TGLSelectRecord;
class TGLViewer;

class FWGeometryTableView;
class FWOverlapTableView;
class TBuffer3D;
class TGeoNode;
class FWGeoTopNodeGLScene;
class FWPopupMenu;

class FWGeoTopNode : public TEveElementList,
                     public TAttBBox
{
   friend class FWGeoTopNodeGL;
public:
      
   enum MenuOptions {
      kSetTopNode,
      kSetTopNodeCam,
      kVisSelfOff,
      kVisChldOn,
      kVisChldOff,
      kApplyChldCol,
      kApplyChldColRec,
      kCamera,
      kPrintMaterial,
      kPrintPath,
      kPrintShape,
      kPrintOverlap,
      kOverlapVisibilityMotherOn,
      kOverlapVisibilityMotherOff
   };
   
   FWGeoTopNode(const char* n = "FWGeoTopNode", const char* t = "FWGeoTopNode"){}
   ~FWGeoTopNode() override{}

   void Paint(Option_t* option="") override;
   FWGeoTopNodeGLScene    *m_scene;
   
   virtual FWGeometryTableManagerBase* tableManager() { return nullptr; }
   virtual FWGeometryTableViewBase* browser() { return nullptr; }
   
   std::set<TGLPhysicalShape*> fHted;
   std::set<TGLPhysicalShape*> fSted;

   int getFirstSelectedTableIndex();
   bool selectPhysicalFromTable(int);
   void clearSelection() {fHted.clear(); fSted.clear();}

   void printSelected();
   virtual void popupMenu(int x, int y, TGLViewer*) {}

   void UnSelected() override;
   void UnHighlighted() override;
   
   static TGLVector3 s_pickedCamera3DCenter;
   static TGLViewer* s_pickedViewer;

protected:
   static UInt_t phyID(int tableIdx);
   static int tableIdx(TGLPhysicalShape* ps);

   void ProcessSelection(TGLSelectRecord& rec, std::set<TGLPhysicalShape*>& sset, TGLPhysicalShape* id);

   void EraseFromSet(std::set<TGLPhysicalShape*>& sset, TGLPhysicalShape* id);
   void ClearSet(std::set<TGLPhysicalShape*>& sset);

   void SetStateOf(TGLPhysicalShape* id);


   void setupBuffMtx(TBuffer3D& buff, const TGeoHMatrix& mat);
   
   FWPopupMenu* setPopupMenu(int iX, int iY, TGLViewer* v, bool);


   void paintShape(Int_t idx,  const TGeoHMatrix& nm, bool volumeColor, bool parentNode);
   void ComputeBBox() override;
private:   
   FWGeoTopNode(const FWGeoTopNode&); // stop default
   const FWGeoTopNode& operator=(const FWGeoTopNode&); // stop default
#ifndef __CINT__
   UChar_t wrapTransparency(FWGeometryTableManagerBase::NodeInfo& data, bool leafNode); 
#endif

   
   ClassDefOverride(FWGeoTopNode, 0);
};


#endif
