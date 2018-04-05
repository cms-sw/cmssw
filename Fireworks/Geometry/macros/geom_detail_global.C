// Author: Matevz Tadel
// Shows some muon chamber nodes in global coordinate system.

#include "common_foos.C"

void geom_detail_global()
{
   TEveUtil::LoadMacro("common_foos.C+");
   std_init();

   make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1/muonBase:MB_1/muonBase:MBWheel_1N_2/mb1:MB1N_10",
	     4, kTRUE);

   make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1/muonBase:MB_1/muonBase:MBWheel_1N_2/mb2:MB2N32N_10",
	     4, kTRUE);

   make_node("/cms:World_1/cms:CMSE_1/muonBase:MUON_1/muonBase:MB_1/muonBase:MBWheel_1N_2/muonYoke:YB1_w1N_m6_10",
	     4, kTRUE);

   gEve->FullRedraw3D(kTRUE);
}
