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
// $Id: FWGeoTopNode.cc,v 1.8 2011/07/05 19:18:49 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/interface/FWGeometryBrowser.h"
#include "Fireworks/Core/interface/FWGeometryTableManager.h"

#include "TEveTrans.h"
#include "TEveManager.h"


#include "TROOT.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualViewer3D.h"
#include "TColor.h"

#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoShapeAssembly.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TVirtualGeoPainter.h"

FWGeoTopNode::FWGeoTopNode(FWGeometryBrowser* t):
   m_geoBrowser(t),
   m_maxLevel(0)
{
   m_entries = &(m_geoBrowser->getTableManager()->refEntries());
}

FWGeoTopNode::~FWGeoTopNode()
{
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

//______________________________________________________________________________
void FWGeoTopNode::Paint(Option_t*)
{
   int topIdx = m_geoBrowser->getTableManager()->getTopGeoNodeIdx();
   FWGeometryTableManager::Entries_i sit = m_entries->begin(); 

   m_maxLevel = m_geoBrowser->getVisLevel() + m_geoBrowser->getTableManager()->getLevelOffset() -1;
   m_filterOff = m_geoBrowser->getFilter().empty();
   TGeoHMatrix mtx;
   if (topIdx >= 0)
   {
      std::advance(sit, topIdx);
      {
         // init matrix
         int pIdx = sit->m_parent;
         int level = sit->m_level;
         int* pl = new int[level];
         while (pIdx != -1)
         {
            pl[ m_entries->at(pIdx).m_level] = pIdx;
            pIdx = m_entries->at(pIdx).m_parent;
         }

         for (int i = 0; i < level; ++i ) {
            TGeoNode* node = m_entries->at(pl[i]).m_node;
            // printf("parents %d = %s \n", pl[i], node->GetName() );
            mtx.Multiply(node->GetMatrix());
         }
         delete [] pl;

         mtx.Multiply(sit->m_node->GetMatrix());
      }

      // paint this node
      if (sit->m_node->IsVisible())
      {
         bool draw = true;
         if ( m_filterOff == false) {
            m_geoBrowser->getTableManager()->assertNodeFilterCache(*sit);
            draw = sit->testBit(FWGeometryTableManager::kMatches);
         }

         if (draw)
            paintShape(*sit, mtx);
      }
   }

   if (sit->m_node->IsVisDaughters())
      paintChildNodesRecurse( sit, mtx);
}

// ______________________________________________________________________

void FWGeoTopNode::paintChildNodesRecurse (FWGeometryTableManager::Entries_i pIt, const TGeoHMatrix& parentMtx)
{ 
   TGeoNode* parentNode =  pIt->m_node;
   int nD = parentNode->GetNdaughters();

   int dOff=0;
   pIt++;

   FWGeometryTableManager::Entries_i it;
   for (int n = 0; n != nD; ++n)
   {
      it =  pIt;
      std::advance(it,n + dOff);

      TGeoHMatrix nm = parentMtx;
      nm.Multiply(it->m_node->GetMatrix());

  
      if (m_filterOff)
      {
         if (it->m_node->IsVisible())
            paintShape(*it, nm);

         if (it->m_node->IsVisDaughters() && it->m_level < m_maxLevel )
            paintChildNodesRecurse(it, nm);

      }
      else
      {
         m_geoBrowser->getTableManager()->assertNodeFilterCache(*it);
         if (it->testBit(FWGeometryTableManager::kMatches) )
            paintShape(*it, nm);

         if (it->testBit(FWGeometryTableManager::kChildMatches) )
            paintChildNodesRecurse(it, nm);
      }


      FWGeometryTableManager::getNNodesTotal(parentNode->GetDaughter(n), dOff);  
   }
}
  
// ______________________________________________________________________
void FWGeoTopNode::paintShape(FWGeometryTableManager::NodeInfo& data, const TGeoHMatrix& nm)
{ 
   static const TEveException eh("FWGeoTopNode::paintShape ");
  

   TGeoShape* shape = data.m_node->GetVolume()->GetShape();
   TGeoCompositeShape* compositeShape = dynamic_cast<TGeoCompositeShape*>(shape);
   if (compositeShape)
   {
      // printf("!!!!!!!!!!!!!!!!!!!! composite shape\n");
      Double_t halfLengths[3] = { compositeShape->GetDX(), compositeShape->GetDY(), compositeShape->GetDZ() };

      TBuffer3D buff(TBuffer3DTypes::kComposite);
      buff.fID           = data.m_node->GetVolume();
      buff.fColor        = data.m_color;
      buff.fTransparency = data.m_node->GetVolume()->GetTransparency(); 

      nm.GetHomogenousMatrix(buff.fLocalMaster);        
      // RefMainTrans().SetBuffer3D(buff);
      buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).
      buff.SetAABoundingBox(compositeShape->GetOrigin(), halfLengths);
      buff.SetSectionsValid(TBuffer3D::kCore|TBuffer3D::kBoundingBox);

      Bool_t paintComponents = kTRUE;

      // Start a composite shape, identified by this buffer
      if (TBuffer3D::GetCSLevel() == 0)
         paintComponents = gPad->GetViewer3D()->OpenComposite(buff);

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

   }
   else
   {
      TBuffer3D& buff = (TBuffer3D&) shape->GetBuffer3D (TBuffer3D::kCore, kFALSE);
      setupBuffMtx(buff, nm);
      buff.fID           = data.m_node->GetVolume();
      buff.fColor        = data.m_color;//node->GetVolume()->GetLineColor();
      buff.fTransparency =  data.m_node->GetVolume()->GetTransparency();

      nm.GetHomogenousMatrix(buff.fLocalMaster);
      buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).

      Int_t sections = TBuffer3D::kBoundingBox | TBuffer3D::kShapeSpecific;
      shape->GetBuffer3D(sections, kTRUE);

           
      Int_t reqSec = gPad->GetViewer3D()->AddObject(buff);

      if (reqSec != TBuffer3D::kNone) {
         // This shouldn't happen, but I suspect it does sometimes.
         if (reqSec & TBuffer3D::kCore)
            Warning(eh, "Core section required again for shape='%s'. This shouldn't happen.", GetName());
         shape->GetBuffer3D(reqSec, kTRUE);
         reqSec = gPad->GetViewer3D()->AddObject(buff);
      }

      if (reqSec != TBuffer3D::kNone)  
         Warning(eh, "Extra section required: reqSec=%d, shape=%s.", reqSec, GetName());
   }
}
