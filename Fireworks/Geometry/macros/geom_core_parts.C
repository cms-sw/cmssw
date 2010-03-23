// @(#)root/eve:$Id: geom_cms.C 31415 2009-11-24 23:31:46Z matevz $
// Author: Matevz Tadel

// Shows CMS tracker, calo and muon system.
// Depth can be set for each part independently.

void geom_core_parts()
{
   TEveUtil::LoadMacro("common_foos.C");
   std_init();

   make_node("/cms:World_1/cms:CMSE_1/tracker:Tracker_1", 4, kTRUE);
   make_node("/cms:World_1/cms:CMSE_1/caloBase:CALO_1",   4, kTRUE);
   make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1",   4, kTRUE);

   gEve->FullRedraw3D(kTRUE);

   std_camera_clip();
}
