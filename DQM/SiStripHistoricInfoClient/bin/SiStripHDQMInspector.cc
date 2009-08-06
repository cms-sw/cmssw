#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/test/HDQMInspectorConfigSiStrip.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

std::string const Condition = "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000";

void SiStripHDQMInspector (const std::string & tagName, std::string const& Password, int const NRuns) {
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
  HDQMInspector A(&StripConfig);
  //HDQMInspector A;
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");


  A.setDebug(1);
  A.setDoStat(1);

  //std::vector<std::string> ItemsForIntegration;
  //ItemsForIntegration.push_back();

  //A.setBlackList("68286");

  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean", "OnTrackClusters.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OffTrack@mean,436207616@Summary_TotalNumberOfClusters_OffTrack@mean,402653184@Summary_TotalNumberOfClusters_OffTrack@mean,469762048@Summary_TotalNumberOfClusters_OffTrack@mean", "TotalNumberOfClusters_OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterChargeCorr_OnTrack@mean,436207616@Summary_ClusterChargeCorr_OnTrack@mean,402653184@Summary_ClusterChargeCorr_OnTrack@mean,469762048@Summary_ClusterChargeCorr_OnTrack@mean", "ClusterChargeCorr_OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterCharge_OffTrack@mean,436207616@Summary_ClusterCharge_OffTrack@mean,402653184@Summary_ClusterCharge_OffTrack@mean,469762048@Summary_ClusterCharge_OffTrack@mean", "ClusterCharge_OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise_OnTrack@mean,436207616@Summary_ClusterNoise_OnTrack@mean,402653184@Summary_ClusterNoise_OnTrack@mean,469762048@Summary_ClusterNoise_OnTrack@mean", "ClusterNoise_OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise_OffTrack@mean,436207616@Summary_ClusterNoise_OffTrack@mean,402653184@Summary_ClusterNoise_OffTrack@mean,469762048@Summary_ClusterNoise_OffTrack@mean", "ClusterNoise_OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoNCorr_OnTrack@mean,436207616@Summary_ClusterStoNCorr_OnTrack@mean,402653184@Summary_ClusterStoNCorr_OnTrack@mean,469762048@Summary_ClusterStoNCorr_OnTrack@mean", "ClusterStoNCorr_OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoN_OffTrack@mean,436207616@Summary_ClusterStoN_OffTrack@mean,402653184@Summary_ClusterStoN_OffTrack@mean,469762048@Summary_ClusterStoN_OffTrack@mean", "ClusterStoN_OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth_OnTrack@mean,436207616@Summary_ClusterWidth_OnTrack@mean,402653184@Summary_ClusterWidth_OnTrack@mean,469762048@Summary_ClusterWidth_OnTrack@mean", "ClusterWidth_OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth_OffTrack@mean,436207616@Summary_ClusterWidth_OffTrack@mean,402653184@Summary_ClusterWidth_OffTrack@mean,469762048@Summary_ClusterWidth_OffTrack@mean", "ClusterWidth_OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_TotalNumberOfDigis@mean,436207616@Summary_TotalNumberOfDigis@mean,402653184@Summary_TotalNumberOfDigis@mean,469762048@Summary_TotalNumberOfDigis@mean", "TotalNumberOfDigis.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterChargeCorr__OnTrack@mean,436207616@Summary_ClusterChargeCorr__OnTrack@mean,402653184@Summary_ClusterChargeCorr__OnTrack@mean,469762048@Summary_ClusterChargeCorr__OnTrack@mean", "_ClusterChargeCorr__OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterCharge__OffTrack@mean,436207616@Summary_ClusterCharge__OffTrack@mean,402653184@Summary_ClusterCharge__OffTrack@mean,469762048@Summary_ClusterCharge__OffTrack@mean", "ClusterCharge__OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise__OnTrack@mean,436207616@Summary_ClusterNoise__OnTrack@mean,402653184@Summary_ClusterNoise__OnTrack@mean,469762048@Summary_ClusterNoise__OnTrack@mean", "ClusterNoise__OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise__OffTrack@mean,436207616@Summary_ClusterNoise__OffTrack@mean,402653184@Summary_ClusterNoise__OffTrack@mean,469762048@Summary_ClusterNoise__OffTrack@mean", "ClusterNoise__OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoNCorr__OnTrack@mean,436207616@Summary_ClusterStoNCorr__OnTrack@mean,402653184@Summary_ClusterStoNCorr__OnTrack@mean,469762048@Summary_ClusterStoNCorr__OnTrack@mean", "ClusterStoNCorr__OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoN__OffTrack@mean,436207616@Summary_ClusterStoN__OffTrack@mean,402653184@Summary_ClusterStoN__OffTrack@mean,469762048@Summary_ClusterStoN__OffTrack@mean", "ClusterStoN__OffTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth__OnTrack@mean,436207616@Summary_ClusterWidth__OnTrack@mean,402653184@Summary_ClusterWidth__OnTrack@mean,469762048@Summary_ClusterWidth__OnTrack@mean", "ClusterWidth__OnTrack.gif", 0, Condition, NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth__OffTrack@mean,436207616@Summary_ClusterWidth__OffTrack@mean,402653184@Summary_ClusterWidth__OffTrack@mean,469762048@Summary_ClusterWidth__OffTrack@mean", "ClusterWidth__OffTrack.gif", 0, Condition, NRuns);

  A.closeFile();


  return;


}


void SiStripHDQMInspector (const std::string &tagName, std::string const& Password, int const Start, int const End) {
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
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");

  A.createTrend("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean", "OnTrackClusters.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_TotalNumberOfClusters_OffTrack@mean,436207616@Summary_TotalNumberOfClusters_OffTrack@mean,402653184@Summary_TotalNumberOfClusters_OffTrack@mean,469762048@Summary_TotalNumberOfClusters_OffTrack@mean", "TotalNumberOfClusters_OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterChargeCorr_OnTrack@mean,436207616@Summary_ClusterChargeCorr_OnTrack@mean,402653184@Summary_ClusterChargeCorr_OnTrack@mean,469762048@Summary_ClusterChargeCorr_OnTrack@mean", "ClusterChargeCorr_OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterCharge_OffTrack@mean,436207616@Summary_ClusterCharge_OffTrack@mean,402653184@Summary_ClusterCharge_OffTrack@mean,469762048@Summary_ClusterCharge_OffTrack@mean", "ClusterCharge_OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterNoise_OnTrack@mean,436207616@Summary_ClusterNoise_OnTrack@mean,402653184@Summary_ClusterNoise_OnTrack@mean,469762048@Summary_ClusterNoise_OnTrack@mean", "ClusterNoise_OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterNoise_OffTrack@mean,436207616@Summary_ClusterNoise_OffTrack@mean,402653184@Summary_ClusterNoise_OffTrack@mean,469762048@Summary_ClusterNoise_OffTrack@mean", "ClusterNoise_OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterStoNCorr_OnTrack@mean,436207616@Summary_ClusterStoNCorr_OnTrack@mean,402653184@Summary_ClusterStoNCorr_OnTrack@mean,469762048@Summary_ClusterStoNCorr_OnTrack@mean", "ClusterStoNCorr_OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterStoN_OffTrack@mean,436207616@Summary_ClusterStoN_OffTrack@mean,402653184@Summary_ClusterStoN_OffTrack@mean,469762048@Summary_ClusterStoN_OffTrack@mean", "ClusterStoN_OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterWidth_OnTrack@mean,436207616@Summary_ClusterWidth_OnTrack@mean,402653184@Summary_ClusterWidth_OnTrack@mean,469762048@Summary_ClusterWidth_OnTrack@mean", "ClusterWidth_OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterWidth_OffTrack@mean,436207616@Summary_ClusterWidth_OffTrack@mean,402653184@Summary_ClusterWidth_OffTrack@mean,469762048@Summary_ClusterWidth_OffTrack@mean", "ClusterWidth_OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_TotalNumberOfDigis@mean,436207616@Summary_TotalNumberOfDigis@mean,402653184@Summary_TotalNumberOfDigis@mean,469762048@Summary_TotalNumberOfDigis@mean", "TotalNumberOfDigis.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterChargeCorr__OnTrack@mean,436207616@Summary_ClusterChargeCorr__OnTrack@mean,402653184@Summary_ClusterChargeCorr__OnTrack@mean,469762048@Summary_ClusterChargeCorr__OnTrack@mean", "_ClusterChargeCorr__OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterCharge__OffTrack@mean,436207616@Summary_ClusterCharge__OffTrack@mean,402653184@Summary_ClusterCharge__OffTrack@mean,469762048@Summary_ClusterCharge__OffTrack@mean", "ClusterCharge__OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterNoise__OnTrack@mean,436207616@Summary_ClusterNoise__OnTrack@mean,402653184@Summary_ClusterNoise__OnTrack@mean,469762048@Summary_ClusterNoise__OnTrack@mean", "ClusterNoise__OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterNoise__OffTrack@mean,436207616@Summary_ClusterNoise__OffTrack@mean,402653184@Summary_ClusterNoise__OffTrack@mean,469762048@Summary_ClusterNoise__OffTrack@mean", "ClusterNoise__OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterStoNCorr__OnTrack@mean,436207616@Summary_ClusterStoNCorr__OnTrack@mean,402653184@Summary_ClusterStoNCorr__OnTrack@mean,469762048@Summary_ClusterStoNCorr__OnTrack@mean", "ClusterStoNCorr__OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterStoN__OffTrack@mean,436207616@Summary_ClusterStoN__OffTrack@mean,402653184@Summary_ClusterStoN__OffTrack@mean,469762048@Summary_ClusterStoN__OffTrack@mean", "ClusterStoN__OffTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterWidth__OnTrack@mean,436207616@Summary_ClusterWidth__OnTrack@mean,402653184@Summary_ClusterWidth__OnTrack@mean,469762048@Summary_ClusterWidth__OnTrack@mean", "ClusterWidth__OnTrack.gif", 0, Condition, Start, End);
  A.createTrend("369098752@Summary_ClusterWidth__OffTrack@mean,436207616@Summary_ClusterWidth__OffTrack@mean,402653184@Summary_ClusterWidth__OffTrack@mean,469762048@Summary_ClusterWidth__OffTrack@mean", "ClusterWidth__OffTrack.gif", 0, Condition, Start, End);


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");




  A.closeFile();


  return;


}






int main (int argc, char* argv[])
{
  if (argc != 4 && argc != 5) {
    std::cerr << "Usage: " << argv[0] << " [TagName] [Password] [NRuns] " << std::endl;
    std::cerr << "Or:    " << argv[0] << " [TagName] [Password] [FirstRun] [LastRun] " << std::endl;
    return 1;
  }

  if (argc == 4) {
    std::cout << "Creating trends for NRuns = " << argv[3] << " for tag: " << argv[1] << std::endl;
    SiStripHDQMInspector( argv[1], argv[2], atoi(argv[3]) );
  } else if(argc == 5) {
    std::cout << "Creating trends for range:  " << argv[3] << " " << argv[4] << " for tag: " << argv[1] << std::endl;
    SiStripHDQMInspector( argv[1], argv[2], atoi(argv[3]), atoi(argv[4]) );
  }

  return 0;
}
