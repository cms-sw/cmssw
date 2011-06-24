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
// $Id$
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
   m_geoBrowser(t)
{
   ComputeBBox();
}

FWGeoTopNode::~FWGeoTopNode()
{
}

int ns = 0;
//______________________________________________________________________________
void FWGeoTopNode::Paint(Option_t*)
{
   ns = 0;
   // printf("FWGeoTopNode::Paint \n");
   TGeoHMatrix mtx;
   paintChildNodesRecurse(-1, mtx, true);
   //printf("paint end %d shapes %d \n", (int)m_geoBrowser->getTableManager()->refEntries().size(), ns);

}

//______________________________________________________________________________
void FWGeoTopNode::ComputeBBox()
{
   BBoxInit();
   float a  =100;
   BBoxCheckPoint(a,a,a);
   BBoxCheckPoint(-a, -a, -a);
}


//______________________________________________________________________________
void FWGeoTopNode::setupBuffMtx(TBuffer3D& buff, TGeoHMatrix& mat)
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
void FWGeoTopNode::paintChildNodesRecurse(int idx, TGeoHMatrix& parentMtx, bool visDaughters)
{
   static const TEveException eh("TEveGeoShape::Paint ");
   FWGeometryTableManager::Entries_v& entries = m_geoBrowser->getTableManager()->refEntries();
   int size = entries.size();


   for (int i=0; i < size; ++i)
   {
      if (entries[i].m_parent == idx )
      {
         TGeoNode* node = entries[i].m_node;
         TGeoHMatrix nm = parentMtx;
         nm.Multiply(node->GetMatrix());

         if (entries[i].m_level >m_geoBrowser->getVisLevel() ) break;

         TGeoShape* shape = node->GetVolume()->GetShape();
         TGeoCompositeShape* compositeShape = dynamic_cast<TGeoCompositeShape*>(shape);
         if (compositeShape)
         {
            // printf("%s has composite shape \n", (*i).name());
         }
         else if (entries[i].m_node->IsVisible() && visDaughters)
         {
            TBuffer3D& buff = (TBuffer3D&) shape->GetBuffer3D (TBuffer3D::kCore, kFALSE);
            setupBuffMtx(buff, nm);
            buff.fID           = node->GetVolume();
            buff.fColor        = node->GetVolume()->GetLineColor();
            buff.fTransparency =  node->GetVolume()->GetTransparency();
            nm.GetHomogenousMatrix(buff.fLocalMaster);
            buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).

            Int_t sections = TBuffer3D::kBoundingBox | TBuffer3D::kShapeSpecific;
            shape->GetBuffer3D(sections, kTRUE);

           
            Int_t reqSec = gPad->GetViewer3D()->AddObject(buff);
            ns++;

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
         paintChildNodesRecurse(i, nm, visDaughters && node->IsVisDaughters());
      }
   }
}

