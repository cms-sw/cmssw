// Shows CMS tracker and PLT
// Depth can be set for each part independently.

#include "common_foos.C"

void geom_plt()
{
   TEveUtil::LoadMacro("common_foos.C+");
   std_init();

   TEveGeoTopNode *plt1, *plt2;

   make_node("/cms:World_1/cms:CMSE_1/tracker:Tracker_1", 4, kTRUE);
   plt1 = make_node("/cms:World_1/cms:CMSE_1/forward:PLT_1", 4, kTRUE);
   plt2 = make_node("/cms:World_1/cms:CMSE_1/forward:PLT_2", 4, kTRUE);

   plt1->SetMainColor(kRed);
   plt1->SetMainTransparency(0);

   gEve->FullRedraw3D(kTRUE);

   std_camera_clip();
}
