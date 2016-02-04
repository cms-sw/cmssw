{
//---------------------------------------------------------------------------------------------------
//
// Creates 2D historic DQM plots
// One HistoricInspector is created per tag. 
// vdetId, vlistItems should be the same for the rested trends.
//
//---------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <time.h>
#include "vector"


//---------------------------------------------------------------------------------------------------

gROOT->Reset();

//---------------------------------------------------------------------------------------------------

std::vector<unsigned int> vRun;
std::vector<float> vSummary;
std::vector<unsigned int> tagInfos;
std::vector<std::string> vlistItems;
std::vector<unsigned int> vdetId;
 
//---------------------------------------------------------------------------------------------------

HistoricInspector A;
A.setDB("oracle://devdb10/CMS_COND_STRIP","historicFromT0_expert","cms_cond_strip","","COND/Services/TBufferBlobStreamingService");
A.setDebug(1);
A.setDoStat(1);
A.setBlackList("69850,69531,69788,69892,69874,69365,69797,69800,69573,68286,69286,69522,66480,69256,67114,67033,67038,67049,67085,67039,69256,67114,67033,67038,67049,67085,67039,69253,65941,65945,65947,65948,65956,65943");

//---------------------------------------------------------------------------------------------------

HistoricInspector B;
B.setDB("oracle://devdb10/CMS_COND_STRIP","historicFromT0_V4_ReReco","cms_cond_strip","QG89MCVZ","COND/Services/TBufferBlobStreamingService");
B.setDebug(1);
B.setDoStat(1);
B.setBlackList("69850,69531,69788,69892,69874,69365,69797,69800,69573,68286,69286,69522,66480,69256,67114,67033,67038,67049,67085,67039,69256,67114,67033,67038,67049,67085,67039,69253,65941,65945,65947,65948,65956,65943");

//---------------------------------------------------------------------------------------------------

TwoDTrends C;

//---------------------------------------------------------------------------------------------------


  //---------------------------------------------------------------------------------------------------
  // Trends for CKF tracks
  //---------------------------------------------------------------------------------------------------

  A.createTrendLastRuns("0@NumberOfTracks_CKFTk@entries,0@NumberOfTracks_CKFTk@mean,0@NumberOfRecHitsPerTrack_CKFTk@mean,0@Chi2_CKFTk@mean,0@Chi2_CKFTk@entries","CKFTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",100);
   
  vRun       = A.getRuns();
  vSummary   = A.getSummary();
  vdetId     = A.getvDetId();
  vlistItems = A.getListItems();
  
  tagInfos.push_back(vSummary.size());

  //---------------------------------------------------------------------------------------------------
  
  B.createTrendLastRuns("0@NumberOfTracks_CKFTk@entries,0@NumberOfTracks_CKFTk@mean,0@NumberOfRecHitsPerTrack_CKFTk@mean,0@Chi2_CKFTk@mean,0@Chi2_CKFTk@entries","CKFTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",100);
  
  // for some ugly reason insert doesn't work ...
  for(size_t j=0;j<B.getRuns().size();++j)     { vRun.push_back(B.getRuns().at(j));}
  for(size_t j=0;j<B.getSummary().size();++j)  { vSummary.push_back(B.getSummary().at(j));}
  
  tagInfos.push_back(vSummary.size());

  //---------------------------------------------------------------------------------------------------
  
  C.twoDplot(vRun, vSummary, vdetId, vlistItems, tagInfos, "CKFTk_trends_2D.gif", 0, 5);

  //---------------------------------------------------------------------------------------------------
 
 
 
  vRun	    .clear();
  vSummary  .clear();
  vdetId    .clear();
  vlistItems.clear();
  tagInfos  .clear();
 
 
 
  
  //---------------------------------------------------------------------------------------------------
  // Trends for CKF tracks - trend vs processing version for a given run
  //---------------------------------------------------------------------------------------------------
  
  
 
 
A.createTrend("0@NumberOfTracks_CKFTk@entries,0@NumberOfTracks_CKFTk@mean,0@NumberOfRecHitsPerTrack_CKFTk@mean,0@Chi2_CKFTk@mean,0@Chi2_CKFTk@entries","CKFTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",70674,70675);
   
  vSummary   = A.getSummary();
  vdetId     = A.getvDetId();
  vlistItems = A.getListItems();
  
 
  //---------------------------------------------------------------------------------------------------
  
 
B.createTrend("0@NumberOfTracks_CKFTk@entries,0@NumberOfTracks_CKFTk@mean,0@NumberOfRecHitsPerTrack_CKFTk@mean,0@Chi2_CKFTk@mean,0@Chi2_CKFTk@entries","CKFTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",70674,70675);
  
  // for some ugly reason insert doesn't work ...
  std::vector<float> vSummary2;
  
  if ( A.getSummary().size() == B.getSummary().size())
  {
    for(size_t j=0;j<A.getSummary().size()/2;++j) {vSummary2.push_back(A.getSummary().at(j));} //temporary
    for(size_t j=0;j<A.getSummary().size()/2;++j) {vSummary2.push_back(B.getSummary().at(j));} //temporary
    C.oneDplot(70674, vSummary, vdetId, vlistItems, 2, "CKFTk_trends_run.gif", 0, 5);}
  else 
  {
    for(size_t j=0;j<A.getSummary().size();++j)  
    { 
     std::cout <<"2 tags don't have the same entries. Is one run missing for one tag ? "<< std::endl;}}
  
 
  //---------------------------------------------------------------------------------------------------
  

}
