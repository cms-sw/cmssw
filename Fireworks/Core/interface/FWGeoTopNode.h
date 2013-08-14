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
// $Id: FWGeoTopNode.h,v 1.13 2012/05/08 02:32:51 amraktad Exp $
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
      kCamera,
      kPrintMaterial,
      kPrintPath,
      kPrintShape,
      kPrintOverlap,
      kOverlapVisibilityMotherOn,
      kOverlapVisibilityMotherOff
   };
   
   FWGeoTopNode(const char* n = "FWGeoTopNode", const char* t = "FWGeoTopNode"){}
   virtual ~FWGeoTopNode(){}

   virtual void Paint(Option_t* option="");
   FWGeoTopNodeGLScene    *m_scene;
   
   virtual FWGeometryTableManagerBase* tableManager() { return 0; }
   virtual FWGeometryTableViewBase* browser() { return 0; }
   
   std::set<TGLPhysicalShape*> fHted;
   std::set<TGLPhysicalShape*> fSted;

   int getFirstSelectedTableIndex();
   bool selectPhysicalFromTable(int);
   void clearSelection() {fHted.clear(); fSted.clear();}

   void printSelected();
   virtual void popupMenu(int x, int y, TGLViewer*) {}

   virtual void UnSelected();
   virtual void UnHighlighted();
   
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
   virtual void ComputeBBox();
private:   
   FWGeoTopNode(const FWGeoTopNode&); // stop default
   const FWGeoTopNode& operator=(const FWGeoTopNode&); // stop default
#ifndef __CINT__
   UChar_t wrapTransparency(FWGeometryTableManagerBase::NodeInfo& data, bool leafNode); 
#endif

   
   ClassDef(FWGeoTopNode, 0);
};


#endif
