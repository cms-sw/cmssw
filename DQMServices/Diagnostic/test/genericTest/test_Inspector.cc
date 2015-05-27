{
#include <stdio.h>
#include <time.h>


/* 
   Extraction of the summary informations using 
   DQMServices/Diagnostic/test/HDQMInspector.
   The sqlite database should have been filled using the new SiStripHistoryDQMService.   
   
   */

gSystem->Load("libFWCoreFWLite");  

FWLiteEnabler::enable();
   
gROOT->Reset();


HDQMInspector A;
A.setDB("sqlite_file:dbfile.db","HDQM_test","cms_cond_strip","w3807dev","");

A.setDebug(1);
A.setDoStat(1);

//A.setBlackList("68286");


//createTrend(std::string ListItems, std::string CanvasName="", int logy=0,std::string Conditions="", unsigned int firstRun=1, unsigned int lastRun=0xFFFFFFFE);
A.createTrend("
1@StaMuon_p@entries,1@StaMuon_p@mean,
1@StaMuon_pt@entries,1@StaMuon_pt@mean,
1@StaMuon_q@entries,1@StaMuon_q@mean,1@StaMuon_q@userExample_XMax,
1@StaMuon_eta@entries,1@StaMuon_eta@mean,
1@StaMuon_theta@entries,1@StaMuon_theta@mean,
1@StaMuon_phi@entries,1@StaMuon_phi@mean
",
"StaMuon.gif",0,"");


}
