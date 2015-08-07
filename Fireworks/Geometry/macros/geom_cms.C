// Author: Matevz Tadel
// Shows full CMS geometry.

#include "common_foos.C"

TEveGeoTopNode *g_cms_all = 0;

void geom_cms()
{
   TEveUtil::LoadMacro("common_foos.C+");
   std_init();

   g_cms_all = make_node("/cms:World_1/cms:CMSE_1", 4, kTRUE);

   gEve->FullRedraw3D(kTRUE);

   std_camera_clip();
}
