#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveScene.h"
#include "TEveGeoNode.h"
#include "TEveTrans.h"

#include "TGLUtil.h"
#include "TGLViewer.h"
#include "TGLCamera.h"
#include "TGLClip.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoMatrix.h"

#include "TGListTree.h"

#include "TPRegexp.h"

typedef TGLClip::EType EClipType;

//==============================================================================
// Creation, initialization
//==============================================================================

void std_init()
{
   TEveManager::Create();
   gGeoManager = gEve->GetGeometry("cmsSimGeom-14.root");
   gGeoManager->DefaultColors();
}

TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs)
{
  if (! gGeoManager->cd(path))
  {
    Warning("make_node", "Path '%s' not found.", path.Data());
    return 0;
  }

  TEveGeoTopNode* tn = new TEveGeoTopNode(gGeoManager, gGeoManager->GetCurrentNode());
  tn->SetVisLevel(vis_level);
  if (global_cs)
  {
    tn->RefMainTrans().SetFrom(*gGeoManager->GetCurrentMatrix());
  }
  gEve->AddGlobalElement(tn);

  return tn;
}

void std_camera_clip()
{
   // EClipType not exported to CINT (see TGLUtil.h):
   // 0 - no clip, 1 - clip plane, 2 - clip box

   TGLViewer *v = gEve->GetDefaultGLViewer();
   v->GetClipSet()->SetClipType((EClipType) 1);
   v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0);
   v->RefreshPadEditor(v);

   v->CurrentCamera().RotateRad(-1.2, 0.5);
   v->DoDraw();
}


//==============================================================================
// EVE helpers
//==============================================================================

void update_evegeonodes(TEveElement* el, Bool_t top)
{
   TEveGeoNode *en = dynamic_cast<TEveGeoNode*>(el);

   if (en && !top)
   {
      TGeoNode *n = en->GetNode();
      en->SetRnrSelfChildren(n->IsVisible(), n->IsVisDaughters());
      en->SetMainColor(n->GetVolume()->GetLineColor());
      en->SetMainTransparency(n->GetVolume()->GetTransparency());
   }

   for (TEveElement::List_i i = el->BeginChildren(); i != el->EndChildren(); ++i)
   {
      update_evegeonodes(*i, kFALSE);
   }
}

void full_update()
{
   TEveScene *gs = gEve->GetGlobalScene();
   for (TEveElement::List_i i = gs->BeginChildren(); i != gs->EndChildren(); ++i)
   {
      update_evegeonodes(*i, kTRUE);
   }
   gEve->FullRedraw3D();
   gEve->GetListTree()->ClearViewPort();
}


//==============================================================================
// Global node / volume toggles
//==============================================================================

void visibility_all_volumes(Bool_t vis_state)
{
   TGeoVolume *v;
   TIter it(gGeoManager->GetListOfVolumes());
   while ((v = (TGeoVolume*) it()) != 0)
   {
      v->SetVisibility(vis_state);
   }
   full_update();
}

void visibility_all_nodes(Bool_t vis_state)
{
   TGeoVolume *v;
   TIter it(gGeoManager->GetListOfVolumes());
   while ((v = (TGeoVolume*) it()) != 0)
   {
      TGeoNode *n;
      TIter it2(v->GetNodes());
      while ((n = (TGeoNode*) it2()) != 0)
      {
         n->SetVisibility(vis_state);
      }
   }
   full_update();
}


//==============================================================================
// Utilities by material type
//==============================================================================

// List materials:
//   gGeoManager->GetListOfMaterials()->ls()
//   gGeoManager->GetListOfMaterials()->ls("*Si*")
//
// Print materials:
//   gGeoManager->GetListOfMaterials()->Print()
//   gGeoManager->GetListOfMaterials()->Print("", "*Si*")

void set_volume_color_by_material(const char* material_re, Color_t color, Char_t transparency=-1)
{
   // Note: material_re is a perl regexp!
   // If you want exact match, enclose in begin / end meta characters (^ / $):
   //   set_volume_color_by_material("^materials:Silicon$", kRed);

   TPMERegexp re(material_re, "o");
   TGeoMaterial *m;
   TIter it(gGeoManager->GetListOfMaterials());
   while ((m = (TGeoMaterial*) it()) != 0)
   {
      if (re.Match(m->GetName()))
      {
         if (transparency != -1)
         {
            m->SetTransparency(transparency);
         }
         TGeoVolume *v;
         TIter it2(gGeoManager->GetListOfVolumes());
         while ((v = (TGeoVolume*) it2()) != 0)
         {
            if (v->GetMaterial() == m)
            {
               v->SetLineColor(color);
            }
         }
      }
   }
   full_update();
}

void visibility_volume_by_material(const char* material_re, Bool_t vis_state)
{
   TPMERegexp re(material_re, "o");
   TGeoMaterial *m;
   TIter it(gGeoManager->GetListOfMaterials());
   while ((m = (TGeoMaterial*) it()) != 0)
   {
      if (re.Match(m->GetName()))
      {
         TGeoVolume *v;
         TIter it2(gGeoManager->GetListOfVolumes());
         while ((v = (TGeoVolume*) it2()) != 0)
         {
            if (v->GetMaterial() == m)
            {
               v->SetVisibility(vis_state);
            }
         }
      }
   }
   full_update();
}
