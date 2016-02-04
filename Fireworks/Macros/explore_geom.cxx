{
   em = TEveManager::Create();
   gGeoManager = em->GetGeometry("cmsGeom10.root");
   TEveGeoTopNode* topn_re = new TEveGeoTopNode(gGeoManager, gGeoManager->GetTopNode());
   gEve->AddGlobalElement(topn_re);
   gEve->Redraw3D();
}
