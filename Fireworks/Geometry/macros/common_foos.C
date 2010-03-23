void std_init()
{
   TEveManager::Create();
   gGeoManager = gEve->GetGeometry("cmsGeom11.root");
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
}

void std_camera_clip()
{
   // EClipType not exported to CINT (see TGLUtil.h):
   // 0 - no clip, 1 - clip plane, 2 - clip box

   TGLViewer *v = gEve->GetDefaultGLViewer();
   v->GetClipSet()->SetClipType(1);
   v->SetGuideState(TGLUtil::kAxesEdge, kTRUE, kFALSE, 0);
   v->RefreshPadEditor(v);

   v->CurrentCamera().RotateRad(-1.2, 0.5);
   v->DoDraw();
}
