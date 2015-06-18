// Author: Matevz Tadel
// Shows one muon chamber in local coordinate system.

#include "common_foos.C"

void geom_detail_local()
{
   TEveUtil::LoadMacro("common_foos.C+");
   std_init();

   make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1/muonBase:MB_1/muonBase:MBWheel_1N_2/mb1:MB1N_10",
	     4, kFALSE);

   gEve->FullRedraw3D(kTRUE);
}
