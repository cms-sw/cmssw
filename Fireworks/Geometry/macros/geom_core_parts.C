// Author: Matevz Tadel

// Shows CMS tracker, calo and muon system.
// Depth can be set for each part independently.
#include "common_foos.C"

TEveGeoTopNode *g_cms_tracker = 0;
TEveGeoTopNode *g_cms_calo    = 0;
TEveGeoTopNode *g_cms_muon    = 0;

void geom_core_parts()
{
   TEveUtil::LoadMacro("common_foos.C+");
   std_init();

   g_cms_tracker = make_node("/cms:World_1/cms:CMSE_1/tracker:Tracker_1", 4, kTRUE);
   g_cms_calo    = make_node("/cms:World_1/cms:CMSE_1/caloBase:CALO_1",   4, kTRUE);
   g_cms_muon    = make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1",   4, kTRUE);

   gEve->FullRedraw3D(kTRUE);

   std_camera_clip();
}
