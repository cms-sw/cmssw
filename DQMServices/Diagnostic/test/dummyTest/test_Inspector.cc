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
// DB, Tag, User, Password, ...
// A.setDB("sqlite_file:dbfile.db","HDQM_HDQM_SiStrip_V1","cms_cond_strip","w3807dev","");
A.setDB("sqlite_file:dbfile.db","HDQM_V1_SiStrip","cms_cond_strip","w3807dev","");

A.setDebug(1);
A.setDoStat(1);

//A.setBlackList("68286");


//createTrend(std::string ListItems, std::string CanvasName="", int logy=0,std::string Conditions="", unsigned int firstRun=1, unsigned int lastRun=0xFFFFFFFE);
//A.createTrend("268435456@Chi2_CKFTk@entries", "number_of_tracks.gif",1);//,"268435456@NumberOfTracks_CKFTk@entries>10000&&268435456@NumberOfRecHitsPerTrack_CKFTk@entries>0");



A.createTrend("
369098752@Summary_TotalNumberOfClusters_OnTrack@mean,
436207616@Summary_TotalNumberOfClusters_OnTrack@mean,
402653184@Summary_TotalNumberOfClusters_OnTrack@mean,
469762048@Summary_TotalNumberOfClusters_OnTrack@mean",
	      "OnTrackClusters.gif",1);//,"268435456@NumberOfTracks_CKFTk@entries>10000&&268435456@NumberOfRecHitsPerTrack_CKFTk@entries>0");


}
