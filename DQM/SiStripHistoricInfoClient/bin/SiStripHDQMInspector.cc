#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/test/HDQMInspectorConfigSiStrip.h"
#include <string>

void SiStripHDQMInspector (const std::string & tagName, int const NRuns) {
/////////////////////////////////////////////////////////////////
//
// Extraction of the summary information using 
// DQMServices/Diagnostic/test/HDQMInspector.
// The sqlite database should have been filled using the new
// SiStripHistoryDQMService.   
//
/////////////////////////////////////////////////////////////////


  //std::map<int, std::string> pixelTranslator = sipixelsummary::GetMap();

  //pixelTranslator Translator;

  //AutoLibraryLoader::enable();

  HDQMInspectorConfigSiStrip StripConfig;
  //HDQMInspector A(&StripConfig);
  HDQMInspector A;
  A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");

  //A.createTrend("268435456@Chi2_CKFTk@entries", "number_of_tracks.gif",1,"268435456@NumberOfTracks_CKFTk@entries>10000&&268435456@NumberOfRecHitsPerTrack_CKFTk@entries>0",NRuns);

  // 
  //A.createTrend("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean",
  //                       "OnTrackClusters.gif",0,
  //                       "268435456@NumberOfTracks_CKFTk@entries>10000&&268435456@NumberOfRecHitsPerTrack_CKFTk@entries>0",NRuns);


A.createTrend("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean",
            "OnTrackClusters.gif",1);



  A.closeFile();


  return;


}


void SiStripHDQMInspector (const std::string &tagName, int const Start, int const End) {
/////////////////////////////////////////////////////////////////
//
// Extraction of the summary information using 
// DQMServices/Diagnostic/test/HDQMInspector.
// The sqlite database should have been filled using the new
// SiStripHistoryDQMService.   
//
/////////////////////////////////////////////////////////////////


  //std::map<int, std::string> pixelTranslator = sipixelsummary::GetMap();

  //pixelTranslator Translator;

  //AutoLibraryLoader::enable();

  HDQMInspectorConfigSiStrip StripConfig;
  //HDQMInspector A(&StripConfig);
  HDQMInspector A(&StripConfig);
  A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");

  A.createTrend("1@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_charge_OffTrack@yMean", "charge_OffTrack_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_charge_OnTrack@yMean", "charge_OnTrack_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_nclusters_OffTrack@yMean", "nclusters_OffTrack_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_nclusters_OnTrack@yMean", "nclusters_OnTrack_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_size_OffTrack@yMean", "size_OffTrack_yMean.gif", 0, "", Start, End);
  A.createTrend("1@SUMOFF_size_OnTrack@yMean", "size_OnTrack_yMean.gif", 0, "", Start, End);


  A.closeFile();


  return;


}






int main (int argc, char* argv[])
{
  if (argc != 3 && argc != 4) {
    std::cerr << "Usage: " << argv[0] << " [TagName] [NRuns] " << std::endl;
    std::cerr << "Or:    " << argv[0] << " [TagName] [FirstRun] [LastRun] " << std::endl;
    return 1;
  }

  if (argc == 3) {
    std::cout << "Creating trends for NRuns = " << argv[2] << " for tag: " << argv[1] << std::endl;
    SiStripHDQMInspector( argv[1], atoi(argv[2]) );
  } else if(argc == 4) {
    std::cout << "Creating trends for range:  " << argv[2] << " " << argv[3] << " for tag: " << argv[1] << std::endl;
    SiStripHDQMInspector( argv[1], atoi(argv[2]), atoi(argv[3]) );
  }

  return 0;
}
