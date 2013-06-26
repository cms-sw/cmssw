// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeoTopNode
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Matevz Tadel, Alja Mrak Tadel  
//         Created:  Thu Jun 23 01:24:51 CEST 2011
// $Id: FWGeoTopNode.cc,v 1.25 2012/05/08 02:32:50 amraktad Exp $
//

// system include files

// user include files

#include "TEveTrans.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveUtil.h"


#include "TROOT.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualViewer3D.h"
#include "TColor.h"
#include "TGLScenePad.h"
#include "TGLPhysicalShape.h"
#include "TGLSelectRecord.h"
#include "TGLViewer.h"
#include "TGLWidget.h"

#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoShapeAssembly.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TVirtualGeoPainter.h"

#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/src/FWGeoTopNodeScene.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/fwLog.h"

TGLViewer* FWGeoTopNode::s_pickedViewer = 0;
TGLVector3 FWGeoTopNode::s_pickedCamera3DCenter;

UInt_t FWGeoTopNode::phyID(int tableIdx) 
{
return UInt_t(tableIdx + 2);

}

int FWGeoTopNode::tableIdx(TGLPhysicalShape* ps) 
{
   return ps->ID() - 2;
}
//______________________________________________________________________________
void FWGeoTopNode::EraseFromSet(std::set<TGLPhysicalShape*>& sset, TGLPhysicalShape* id)
{
   sset.erase(id);
   SetStateOf(id);
}

//______________________________________________________________________________
void FWGeoTopNode::ClearSet(std::set<TGLPhysicalShape*>& sset)
{
   while (!sset.empty())
   {
      TGLPhysicalShape *id = *sset.begin();
      sset.erase(id);
      SetStateOf(id);
   }
}
//______________________________________________________________________________
void FWGeoTopNode::SetStateOf(TGLPhysicalShape* id)
{
   FWGeometryTableManagerBase::NodeInfo& data =  tableManager()->refEntries().at(tableIdx(id));

   if (fSted.find(id) != fSted.end())
   {
      id->Select(1);
      data.setBit(FWGeometryTableManagerBase::kSelected);
   }
   else if (fHted.find(id) != fHted.end())
   {
      id->Select(3);
      data.setBit(FWGeometryTableManagerBase::kHighlighted);
   }
   else
   {
      id->Select(0);
      data.resetBit(FWGeometryTableManagerBase::kHighlighted);
      data.resetBit(FWGeometryTableManagerBase::kSelected);
   }

}

//______________________________________________________________________________
void FWGeoTopNode::ProcessSelection(TGLSelectRecord& rec, std::set<TGLPhysicalShape*>& sset, TGLPhysicalShape* id)
{
   // printf("FWGeoTopNode::ProcessSelection ===============================\n");

   m_scene->BeginUpdate();

   if (sset.empty())
   {
      if (id)
      {
         sset.insert(id);
         rec.SetSecSelResult(TGLSelectRecord::kEnteringSelection);
      }  
   }
   else
   {
      if (id)
      {
         if (rec.GetMultiple())
         {
            if (sset.find(id) == sset.end())
            {
               sset.insert(id);
               rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);
            }
            else
            {
               EraseFromSet(sset, id);
               if (sset.empty())
                  rec.SetSecSelResult(TGLSelectRecord::kLeavingSelection);
               else
                  rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);
            }
         }
         else
         {
            if (sset.size() != 1 || sset.find(id) == sset.end())
            {
               ClearSet(sset);
               sset.insert(id);
               rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);
            }
         }
      }
      else
      {
         if (!rec.GetMultiple())
         {
            ClearSet(sset);
            rec.SetSecSelResult(TGLSelectRecord::kLeavingSelection);
         }
      }
   }

   if (id)
   {
      SetStateOf(id);
   }

   if (rec.GetSecSelResult() != TGLSelectRecord::kNone)
   {
      m_scene->EndUpdate(kTRUE, kFALSE, kTRUE);
      gEve->Redraw3D();

      tableManager()->dataChanged();
   }
   else
   {
      m_scene->EndUpdate(kFALSE, kFALSE, kFALSE);
   }
}

//______________________________________________________________________________
bool FWGeoTopNode::selectPhysicalFromTable( int tableIndex)
{
   //   printf("FWGeoTopNode::selectPhysicalFromTable 

   TGLPhysicalShape* ps = m_scene->FindPhysical(phyID(tableIndex));
   if (ps) {
      fSted.insert(ps);
      ps->Select(1);
      // printf("selectPhysicalFromTable found physical \n");
      return true;
   }
   else if ( tableManager()->refEntries().at(tableIndex).testBit(FWGeometryTableManagerBase::kVisNodeSelf));
   {
      fwLog(fwlog::kInfo) << "Selected entry not drawn in GL viewer. \n" ;
      return false;
   }
}

//______________________________________________________________________________
void FWGeoTopNode::printSelected() 
{
   for (std::set<TGLPhysicalShape*>::iterator it = fSted.begin(); it != fSted.end(); ++it)
   {
      printf("FWGeoTopNode::printSelected %s \n",  tableManager()->refEntries().at(tableIdx(*it)).name() );
   }
}

//______________________________________________________________________________

int FWGeoTopNode::getFirstSelectedTableIndex() 
{
   // Note: if object would be rendered, this would return fSted.begin().

   if (fSted.size() <= 1)
   {
      int cnt = 0;
      for (FWGeometryTableManagerBase::Entries_i i = tableManager()->refEntries().begin(); i != tableManager()->refEntries().end(); ++i, ++cnt)
      {
         if (i->testBit(FWGeometryTableManagerBase::kSelected)) return cnt; 
      }
   }
   return -1;
}

//______________________________________________________________________________
void FWGeoTopNode::ComputeBBox()
{
   // Fill bounding-box information. Virtual from TAttBBox.

   BBoxZero(1.0f);
}

//______________________________________________________________________________
void FWGeoTopNode::setupBuffMtx(TBuffer3D& buff, const TGeoHMatrix& mat)
{
   const Double_t *r = mat.GetRotationMatrix();
   const Double_t *t = mat.GetTranslation();
   const Double_t *s = mat.GetScale();
   Double_t       *m = buff.fLocalMaster;
   m[0]  = r[0]*s[0]; m[1]  = r[1]*s[1]; m[2]  = r[2]*s[2]; m[3]  = 0;
   m[4]  = r[3]*s[0]; m[5]  = r[4]*s[1]; m[6]  = r[5]*s[2]; m[7]  = 0;
   m[8]  = r[6]*s[0]; m[9]  = r[7]*s[1]; m[10] = r[8]*s[2]; m[11] = 0;
   m[12] = t[0];      m[13] = t[1];      m[15] = t[2];      m[15] = 1;

   buff.fLocalFrame = kTRUE;
}

// ______________________________________________________________________
void FWGeoTopNode::paintShape( Int_t tableIndex, const TGeoHMatrix& nm, bool volumeColor, bool isParentNode)
{
   static const TEveException eh("FWGeoTopNode::paintShape ");
 
  // printf("paint sahpe id %d\n", tableIndex );

   FWGeometryTableManagerBase::NodeInfo& data = tableManager()->refEntries().at(tableIndex);
   UChar_t transparency = wrapTransparency(data, isParentNode);
   // printf("trans %d \n", transparency );
   if (transparency >= 100) return;
   
   TGeoShape* shape = data.m_node->GetVolume()->GetShape();
   
   TGeoCompositeShape* compositeShape = dynamic_cast<TGeoCompositeShape*>(shape);
   if (compositeShape)
   {
      // m_scene->fNextCompositeID = phyID(tableIndex);

      Double_t halfLengths[3] = { compositeShape->GetDX(), compositeShape->GetDY(), compositeShape->GetDZ() };

      TBuffer3D buff(TBuffer3DTypes::kComposite);
      buff.fID           = data.m_node->GetVolume();
      buff.fColor        = volumeColor ? data.m_node->GetVolume()->GetLineColor() : data.m_color ;
      buff.fTransparency = transparency;// data.m_node->GetVolume()->GetTransparency(); 

      nm.GetHomogenousMatrix(buff.fLocalMaster);  
      buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).
      buff.SetAABoundingBox(compositeShape->GetOrigin(), halfLengths);
      buff.SetSectionsValid(TBuffer3D::kCore|TBuffer3D::kBoundingBox);

      Bool_t paintComponents = kTRUE;
      // Start a composite shape, identified by this buffer
      if (TBuffer3D::GetCSLevel() == 0) {
         paintComponents = m_scene->OpenCompositeWithPhyID(phyID(tableIndex), buff);
      }

      TBuffer3D::IncCSLevel();
      
      // Paint the boolean node - will add more buffers to viewer
      TGeoHMatrix xxx;
      TGeoMatrix *gst = TGeoShape::GetTransform();
      TGeoShape::SetTransform(&xxx);

      if (paintComponents) compositeShape->GetBoolNode()->Paint("");
      TGeoShape::SetTransform(gst);
      // Close the composite shape
      if (TBuffer3D::DecCSLevel() == 0)
         gPad->GetViewer3D()->CloseComposite();


      //  m_scene->fNextCompositeID = 0;
   }
   else
   {
      TBuffer3D& buff = (TBuffer3D&) shape->GetBuffer3D (TBuffer3D::kCore, kFALSE);
      setupBuffMtx(buff, nm);
      buff.fID           = data.m_node->GetVolume();
      buff.fColor        = volumeColor ? data.m_node->GetVolume()->GetLineColor() : data.m_color ;
      buff.fTransparency = transparency;// data.m_node->GetVolume()->GetTransparency(); 


      nm.GetHomogenousMatrix(buff.fLocalMaster);
      buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).

      Int_t sections = TBuffer3D::kBoundingBox | TBuffer3D::kShapeSpecific;
      shape->GetBuffer3D(sections, kTRUE);

      Int_t reqSec = gPad->GetViewer3D()->AddObject(phyID(tableIndex), buff);

      if (reqSec != TBuffer3D::kNone) {
         // This shouldn't happen, but I suspect it does sometimes.
         if (reqSec & TBuffer3D::kCore)
            Warning(eh, "Core section required again for shape='%s'. This shouldn't happen.", GetName());
         shape->GetBuffer3D(reqSec, kTRUE);
         reqSec = gPad->GetViewer3D()->AddObject(phyID(tableIndex), buff);
      }

      if (reqSec != TBuffer3D::kNone)  
         Warning(eh, "Extra section required: reqSec=%d, shape=%s.", reqSec, GetName());
   }
}

// ______________________________________________________________________
void FWGeoTopNode::Paint(Option_t* opt)
{
   static const TEveException eh("FWGeoTopNode::Paint ");

   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = this;
   buff.fColor        = GetMainColor();
   buff.fTransparency = GetMainTransparency();
   if (HasMainTrans())  RefMainTrans().SetBuffer3D(buff);

   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(1, buff);
   if (reqSections != TBuffer3D::kNone)
   {
      Warning(eh, "IsA='%s'. Viewer3D requires more sections (%d). Only direct-rendering supported.",
              ClassName(), reqSections);
   }
}

// ______________________________________________________________________
void FWGeoTopNode::UnSelected()
{
   ClearSet(fSted);
   for (FWGeometryTableManagerBase::Entries_i i = tableManager()->refEntries().begin(); i != tableManager()->refEntries().end(); ++i)
      i->resetBit(FWGeometryTableManagerBase::kSelected);
}

// ______________________________________________________________________
void FWGeoTopNode::UnHighlighted()
{
   ClearSet(fHted);
   for (FWGeometryTableManagerBase::Entries_i i = tableManager()->refEntries().begin(); i != tableManager()->refEntries().end(); ++i)
      i->resetBit(FWGeometryTableManagerBase::kHighlighted);
}   


// ______________________________________________________________________
FWPopupMenu* FWGeoTopNode::setPopupMenu(int iX, int iY, TGLViewer* v, bool overlap)
{
   if (getFirstSelectedTableIndex() < 0)
   {
      if (fSted.empty()) fwLog(fwlog::kInfo) << "No menu -- no node/entry selected \n";
      return 0;
   }
   
   FWPopupMenu* nodePopup = new FWPopupMenu();
   
   nodePopup->AddEntry("Set As Top Node", kSetTopNode);
   nodePopup->AddEntry("Set As Top Node And Reset Camera", kSetTopNodeCam);
   nodePopup->AddSeparator();
   if (v) { 
       nodePopup->AddEntry("Rnr Off", kVisSelfOff);
   }
   nodePopup->AddEntry("Rnr Off For All Children", kVisChldOff);
   nodePopup->AddEntry("Rnr On For All Children", kVisChldOn);      
   nodePopup->AddSeparator();
   
   if (overlap)
      nodePopup->AddEntry("Print Overlap", kPrintOverlap);
   nodePopup->AddEntry("Print Path", kPrintPath);
   nodePopup->AddEntry("Print Shape", kPrintShape);
   nodePopup->AddEntry("Print Material", kPrintMaterial);   
   
   nodePopup->AddSeparator();
   if (v) {
      Window_t wdummy;
      Int_t x,y;
      gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), v->GetGLWidget()->GetId(), iX, iY, x, y, wdummy);   
      TGLVector3 pnt(x, y, 0.5*v->GetSelRec().GetMinZ());
      v->CurrentCamera().WindowToViewport(pnt);
      s_pickedCamera3DCenter = v->CurrentCamera().ViewportToWorld(pnt);
      // s_pickedCamera3DCenter.Dump();
      s_pickedViewer = v;
      
      nodePopup->AddEntry("Set Camera Center", kCamera);
   }
   
   nodePopup->PlaceMenu(iX, iY,true,true);
   return nodePopup;
}


UChar_t FWGeoTopNode::wrapTransparency(FWGeometryTableManagerBase::NodeInfo& data, bool isParentNode)
{
   if (isParentNode)
   {
      return TMath::Max((Char_t)browser()->getMinParentTransparency(), data.m_transparency) * browser()->getParentTransparencyFactor();
   }
   else 
   {
      return TMath::Max((Char_t)browser()->getMinLeafTransparency(), data.m_transparency) * browser()->getLeafTransparencyFactor();
   }   
}
