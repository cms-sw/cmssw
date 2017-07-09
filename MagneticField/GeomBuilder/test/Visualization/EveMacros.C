//
// A collection of Eve macros for geometry visualization.
// Load with .L EveMacros.C+
// For usage, cf:
// https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMagneticField#Visualization_of_the_field_map_i
//

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
#include "TGLSAViewer.h"

#include "TGListTree.h"

#include "TPRegexp.h"

#include <iostream>

//==============================================================================
// Creation, initialization
//==============================================================================

//initialization
void macros(){
  ((TGLSAViewer*)gEve->GetDefaultGLViewer())->DisableMenuBarHiding();
}

// void std_init()
// {
//    TEveManager::Create();
//    gGeoManager = gEve->GetGeometry("cmsSimGeom-14.root");
//    gGeoManager->DefaultColors();
// }

// TEveGeoTopNode* make_node(const TString& path, Int_t vis_level, Bool_t global_cs)
// {
//   if (! gGeoManager->cd(path))
//   {
//     Warning("make_node", "Path '%s' not found.", path.Data());
//     return 0;
//   }

//   TEveGeoTopNode* tn = new TEveGeoTopNode(gGeoManager, gGeoManager->GetCurrentNode());
//   tn->SetVisLevel(vis_level);
//   if (global_cs)
//   {
//     tn->RefMainTrans().SetFrom(*gGeoManager->GetCurrentMatrix());
//   }
//   gEve->AddGlobalElement(tn);

//   return tn;
// }


// void std_camera_clip()
// {
//    // EClipType not exported to CINT (see TGLUtil.h):
//    // 0 - no clip, 1 - clip plane, 2 - clip box

//    TGLViewer *v = gEve->GetDefaultGLViewer();
//    v->GetClipSet()->SetClipType((EClipType) 1);
//    v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0);
//    v->RefreshPadEditor(v);

//    v->CurrentCamera().RotateRad(-1.2, 0.5);
//    v->DoDraw();
// }


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

// Toggle everything on/off
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

// Toggle a specific volume on or off. Volume names are typically:
// MagneticFieldVolumes_1103l:V_1001_1
// FIXME: after running this command, you must click on the toggle box of
// cms::World_1 in the Eve panel. There is no known workaround for this in the macro.
void visibility_all_nodes(Bool_t vis_state, const char* nameFilter=".*")
{

  TPMERegexp re(nameFilter, "o");

   TGeoVolume *v;
   TIter it(gGeoManager->GetListOfVolumes());
   while ((v = (TGeoVolume*) it()) != 0)
   {
      TGeoNode *n;
      TIter it2(v->GetNodes());
      while ((n = (TGeoNode*) it2()) != 0)
      {
	TString name(n->GetName());
	if (name!="cms:World_1"&&name!="cmsMagneticField:MAGF_1" && re.Match(name)) {
	  std::cout << name << std::endl;
	  n->SetVisibility(vis_state);
	 //	 v->SetVisLeaves(vis_state);
	 //v->SetVisContainers(vis_state);
	} else {
	  //	  std::cout << name << std::endl;
	}
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


//==============================================================================
// A collection of predefined views.
// clip=true clips the geometry in a thin section order to draw boundaries on the specified plane.
//==============================================================================

float GetZoom() //FIXME: works only in ORTOGRAPHIC view
{
  TGLViewer* v = gEve->GetDefaultGLViewer();
  TGLOrthoCamera& c = (TGLOrthoCamera& )v->CurrentCamera(); //FIXME should dynamic_cast
  printf("zoom %f \n", c.GetZoom());
  return c.GetZoom();
}

void SetZoom(float zoom) //FIXME: works only in ORTOGRAPHIC view
{
  TGLViewer* v = gEve->GetDefaultGLViewer();
  TGLOrthoCamera& c = (TGLOrthoCamera& )v->CurrentCamera(); //FIXME should dynamic_cast
  c.SetZoom(zoom);
  c.IncTimeStamp();
  v->RequestDraw();
}



void goTo(TString view, bool clip) {

  TGLViewer* v   = gEve->GetDefaultGLViewer();
  TGLClipSet* cs = v->GetClipSet();
  

  if (view=="ZY"){
    v->SetCurrentCamera(TGLViewer::kCameraOrthoZOY);
//     gEve->FullRedraw3D(kTRUE);
//     v->CurrentCamera().RotateRad(0,TMath::Pi()/2.);
//     v->DoDraw();

    if (clip) {
      Double_t clipBox[6] = {0,0,0, 3,1800,4000};
      cs->SetClipType(TGLClip::kClipBox);
      v->RefreshPadEditor(v);
      cs->GetCurrentClip()->SetMode(TGLClip::kOutside);
      cs->SetClipState(TGLClip::kClipBox, clipBox);
      v->RefreshPadEditor(v);
      v->SetStyle(TGLRnrCtx::kOutline);
    }
  }

  else if (view=="XY"){
    v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

    if (clip) {
      Double_t clipBox[6] = {0,0,0, 1800,1800,3};
      cs->SetClipType(TGLClip::kClipBox);
      v->RefreshPadEditor(v);
      cs->GetCurrentClip()->SetMode(TGLClip::kOutside);
      cs->SetClipState(TGLClip::kClipBox, clipBox);
      v->RefreshPadEditor(v);
      v->SetStyle(TGLRnrCtx::kOutline);
    }
    
  }

  gEve->FullRedraw3D(kTRUE);
  //  v->DoDraw();
}




void ZYView(float zoom=-999, bool clip=true){

  TGLViewer* v   = gEve->GetDefaultGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraOrthoZOY);

  if (clip) {
    TGLClipSet* cs = v->GetClipSet();
    Double_t clipBox[6] = {0,0,0, 3,1800,4000};
    cs->SetClipType(TGLClip::kClipBox);
    v->RefreshPadEditor(v);
    cs->GetCurrentClip()->SetMode(TGLClip::kOutside);
    cs->SetClipState(TGLClip::kClipBox, clipBox);
    v->RefreshPadEditor(v);
    v->SetStyle(TGLRnrCtx::kOutline);
  }
  gEve->FullRedraw3D(kTRUE);

  if (zoom>0) SetZoom(zoom);

}


void XYView(float Z=0, bool clip=true){
  TGLViewer* v   = gEve->GetDefaultGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

  if (clip) {
    TGLClipSet* cs = v->GetClipSet();
    Double_t clipBox[6] = {0,0,Z, 1800,1800,3};
    cs->SetClipType(TGLClip::kClipBox);
    v->RefreshPadEditor(v);
    cs->GetCurrentClip()->SetMode(TGLClip::kOutside);
    cs->SetClipState(TGLClip::kClipBox, clipBox);
    v->RefreshPadEditor(v);
    v->SetStyle(TGLRnrCtx::kOutline);
  }
  gEve->FullRedraw3D(kTRUE);    
}

  
void print(TString filename) {
  gEve->GetDefaultGLViewer()->SavePicture(filename);
}



//  Double_t center[3] = {0,0,0};
//    v->SetOrthoCamera(TGLViewer::kCameraOrthoXOY, 0.5, 1500, center, 0,TMath::Pi()/2.);
//    v->ResetCurrentCamera();
//   v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0)


