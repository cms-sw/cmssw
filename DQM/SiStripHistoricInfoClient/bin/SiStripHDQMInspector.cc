#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/test/HDQMInspectorConfigSiStrip.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <vector>

std::string const Condition = "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000";
std::string const BlackList = "109861";

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
  std::vector<std::string> ItemsForIntegration;
  ItemsForIntegration.push_back("TotalNumberOfClusters_OnTrack_entries");
  StripConfig.computeIntegralList(ItemsForIntegration);
  HDQMInspector A(&StripConfig);
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");


  A.setDebug(1);
  A.setDoStat(1);

  //std::vector<std::string> ItemsForIntegration;
  //ItemsForIntegration.push_back();

  A.setBlackList(BlackList);



  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean", "OnTrackClusters.gif", 0, "369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OnTrack@mean > 0", NRuns);

  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OnTrack@entries,436207616@Summary_TotalNumberOfClusters_OnTrack@entries,402653184@Summary_TotalNumberOfClusters_OnTrack@entries,469762048@Summary_TotalNumberOfClusters_OnTrack@entries", "OnTrackClusters_entries.gif", 0, "369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 0", NRuns);

  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OffTrack@mean,436207616@Summary_TotalNumberOfClusters_OffTrack@mean,402653184@Summary_TotalNumberOfClusters_OffTrack@mean,469762048@Summary_TotalNumberOfClusters_OffTrack@mean", "TotalNumberOfClusters_OffTrack.gif", 0, "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OffTrack@mean > 0", NRuns);

  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OffTrack@entries,436207616@Summary_TotalNumberOfClusters_OffTrack@entries,402653184@Summary_TotalNumberOfClusters_OffTrack@entries,469762048@Summary_TotalNumberOfClusters_OffTrack@entries", "TotalNumberOfClusters_OffTrack_entries.gif", 0, "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 0", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterChargeCorr_OnTrack@landauPeak,436207616@Summary_ClusterChargeCorr_OnTrack@landauPeak,402653184@Summary_ClusterChargeCorr_OnTrack@landauPeak,469762048@Summary_ClusterChargeCorr_OnTrack@landauPeak", "ClusterChargeCorr_OnTrack_landau.gif", 0, "369098752@Summary_ClusterChargeCorr_OnTrack@landauPeak > 0 && 369098752@Summary_ClusterChargeCorr_OnTrack@entries > 10000", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterCharge_OffTrack@landauPeak,436207616@Summary_ClusterCharge_OffTrack@landauPeak,402653184@Summary_ClusterCharge_OffTrack@landauPeak,469762048@Summary_ClusterCharge_OffTrack@landauPeak", "ClusterCharge_OffTrack_landau.gif", 0, "369098752@Summary_ClusterCharge_OffTrack@landauPeak > 0 && 369098752@Summary_ClusterCharge_OffTrack@entries > 10000", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterNoise_OnTrack@gaussMean,436207616@Summary_ClusterNoise_OnTrack@gaussMean,402653184@Summary_ClusterNoise_OnTrack@gaussMean,469762048@Summary_ClusterNoise_OnTrack@gaussMean", "ClusterNoise_OnTrack_gauss.gif", 0, "369098752@Summary_ClusterNoise_OnTrack@gaussMean > 0 && 369098752@Summary_ClusterNoise_OnTrack@entries > 10000", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterNoise_OffTrack@gaussMean,436207616@Summary_ClusterNoise_OffTrack@gaussMean,402653184@Summary_ClusterNoise_OffTrack@gaussMean,469762048@Summary_ClusterNoise_OffTrack@gaussMean", "ClusterNoise_OffTrack_gauss.gif", 0, "369098752@Summary_ClusterNoise_OffTrack@gaussMean > 0&& 369098752@Summary_ClusterNoise_OffTrack@entries > 10000", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterStoNCorr_OnTrack@mean,436207616@Summary_ClusterStoNCorr_OnTrack@mean,402653184@Summary_ClusterStoNCorr_OnTrack@mean,469762048@Summary_ClusterStoNCorr_OnTrack@mean", "ClusterStoNCorr_OnTrack.gif", 0, "369098752@Summary_ClusterStoNCorr_OnTrack@entries > 10000 && 369098752@Summary_ClusterStoNCorr_OnTrack@mean > 0", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterStoNCorr_OnTrack@landauPeak,436207616@Summary_ClusterStoNCorr_OnTrack@landauPeak,402653184@Summary_ClusterStoNCorr_OnTrack@landauPeak,469762048@Summary_ClusterStoNCorr_OnTrack@landauPeak", "ClusterStoNCorr_OnTrack_landau.gif", 0, "369098752@Summary_ClusterStoNCorr_OnTrack@landauPeak > 0 && 369098752@Summary_ClusterStoNCorr_OnTrack@entries > 10000", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterStoN_OffTrack@landauPeak,436207616@Summary_ClusterStoN_OffTrack@landauPeak,402653184@Summary_ClusterStoN_OffTrack@landauPeak,469762048@Summary_ClusterStoN_OffTrack@landauPeak", "ClusterStoN_OffTrack_landau.gif", 0, "369098752@Summary_ClusterStoN_OffTrack@landauPeak > 0 && 369098752@Summary_ClusterStoN_OffTrack@entries > 10000", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterWidth_OnTrack@mean,436207616@Summary_ClusterWidth_OnTrack@mean,402653184@Summary_ClusterWidth_OnTrack@mean,469762048@Summary_ClusterWidth_OnTrack@mean", "ClusterWidth_OnTrack.gif", 0, "369098752@Summary_ClusterWidth_OnTrack@entries > 10000 && 369098752@Summary_ClusterWidth_OnTrack@mean > 0", NRuns);

  //A.createTrendLastRuns("369098752@Summary_ClusterStoN_OffTrack@mean,436207616@Summary_ClusterStoN_OffTrack@mean,402653184@Summary_ClusterStoN_OffTrack@mean,469762048@Summary_ClusterStoN_OffTrack@mean", "ClusterStoN_OffTrack.gif", 0, "369098752@Summary_ClusterStoN_OffTrack@entries > 10000 && 369098752@Summary_ClusterStoN_OffTrack@mean > 0", NRuns);
  //A.createTrendLastRuns("369098752@Summary_TotalNumberOfDigis@mean,436207616@Summary_TotalNumberOfDigis@mean,402653184@Summary_TotalNumberOfDigis@mean,469762048@Summary_TotalNumberOfDigis@mean", "TotalNumberOfDigis.gif", 0, "369098752@Summary_TotalNumberOfDigis@entries > 10000 && 369098752@Summary_TotalNumberOfDigis@mean > 0", NRuns);

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
  HDQMInspector A(&StripConfig);
  std::vector<std::string> ItemsForIntegration;
  ItemsForIntegration.push_back("TotalNumberOfClusters_OnTrack_entries");
  StripConfig.computeIntegralList(ItemsForIntegration);
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");


  A.setDebug(1);
  A.setDoStat(1);

  A.setBlackList(BlackList);




  A.createTrend("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean", "OnTrackClusters.gif", 0, "369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OnTrack@mean > 0", Start, End);

  A.createTrend("369098752@Summary_TotalNumberOfClusters_OnTrack@entries,436207616@Summary_TotalNumberOfClusters_OnTrack@entries,402653184@Summary_TotalNumberOfClusters_OnTrack@entries,469762048@Summary_TotalNumberOfClusters_OnTrack@entries", "OnTrackClusters_entries.gif", 0, "369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OnTrack@entries > 0", Start, End);

  A.createTrend("369098752@Summary_TotalNumberOfClusters_OffTrack@mean,436207616@Summary_TotalNumberOfClusters_OffTrack@mean,402653184@Summary_TotalNumberOfClusters_OffTrack@mean,469762048@Summary_TotalNumberOfClusters_OffTrack@mean", "TotalNumberOfClusters_OffTrack.gif", 0, "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OffTrack@mean > 0", Start, End);

  A.createTrend("369098752@Summary_TotalNumberOfClusters_OffTrack@entries,436207616@Summary_TotalNumberOfClusters_OffTrack@entries,402653184@Summary_TotalNumberOfClusters_OffTrack@entries,469762048@Summary_TotalNumberOfClusters_OffTrack@entries", "TotalNumberOfClusters_OffTrack_entries.gif", 0, "369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 10000 && 369098752@Summary_TotalNumberOfClusters_OffTrack@entries > 0", Start, End);

  A.createTrend("369098752@Summary_ClusterChargeCorr_OnTrack@landauPeak,436207616@Summary_ClusterChargeCorr_OnTrack@landauPeak,402653184@Summary_ClusterChargeCorr_OnTrack@landauPeak,469762048@Summary_ClusterChargeCorr_OnTrack@landauPeak", "ClusterChargeCorr_OnTrack_landau.gif", 0, "369098752@Summary_ClusterChargeCorr_OnTrack@landauPeak > 0 && 369098752@Summary_ClusterChargeCorr_OnTrack@entries > 10000", Start, End);

  A.createTrend("369098752@Summary_ClusterCharge_OffTrack@landauPeak,436207616@Summary_ClusterCharge_OffTrack@landauPeak,402653184@Summary_ClusterCharge_OffTrack@landauPeak,469762048@Summary_ClusterCharge_OffTrack@landauPeak", "ClusterCharge_OffTrack_landau.gif", 0, "369098752@Summary_ClusterCharge_OffTrack@landauPeak > 0 && 369098752@Summary_ClusterCharge_OffTrack@entries > 10000", Start, End);

  A.createTrend("369098752@Summary_ClusterNoise_OnTrack@gaussMean,436207616@Summary_ClusterNoise_OnTrack@gaussMean,402653184@Summary_ClusterNoise_OnTrack@gaussMean,469762048@Summary_ClusterNoise_OnTrack@gaussMean", "ClusterNoise_OnTrack_gauss.gif", 0, "369098752@Summary_ClusterNoise_OnTrack@gaussMean > 0 && 369098752@Summary_ClusterNoise_OnTrack@entries > 10000", Start, End);

  A.createTrend("369098752@Summary_ClusterNoise_OffTrack@gaussMean,436207616@Summary_ClusterNoise_OffTrack@gaussMean,402653184@Summary_ClusterNoise_OffTrack@gaussMean,469762048@Summary_ClusterNoise_OffTrack@gaussMean", "ClusterNoise_OffTrack_gauss.gif", 0, "369098752@Summary_ClusterNoise_OffTrack@gaussMean > 0&& 369098752@Summary_ClusterNoise_OffTrack@entries > 10000", Start, End);

  A.createTrend("369098752@Summary_ClusterStoNCorr_OnTrack@mean,436207616@Summary_ClusterStoNCorr_OnTrack@mean,402653184@Summary_ClusterStoNCorr_OnTrack@mean,469762048@Summary_ClusterStoNCorr_OnTrack@mean", "ClusterStoNCorr_OnTrack.gif", 0, "369098752@Summary_ClusterStoNCorr_OnTrack@entries > 10000 && 369098752@Summary_ClusterStoNCorr_OnTrack@mean > 0", Start, End);

  A.createTrend("369098752@Summary_ClusterStoNCorr_OnTrack@landauPeak,436207616@Summary_ClusterStoNCorr_OnTrack@landauPeak,402653184@Summary_ClusterStoNCorr_OnTrack@landauPeak,469762048@Summary_ClusterStoNCorr_OnTrack@landauPeak", "ClusterStoNCorr_OnTrack_landau.gif", 0, "369098752@Summary_ClusterStoNCorr_OnTrack@landauPeak > 0 && 369098752@Summary_ClusterStoNCorr_OnTrack@entries > 10000", Start, End);

  A.createTrend("369098752@Summary_ClusterStoN_OffTrack@landauPeak,436207616@Summary_ClusterStoN_OffTrack@landauPeak,402653184@Summary_ClusterStoN_OffTrack@landauPeak,469762048@Summary_ClusterStoN_OffTrack@landauPeak", "ClusterStoN_OffTrack_landau.gif", 0, "369098752@Summary_ClusterStoN_OffTrack@landauPeak > 0 && 369098752@Summary_ClusterStoN_OffTrack@entries > 10000", Start, End);

  A.createTrend("369098752@Summary_ClusterWidth_OnTrack@mean,436207616@Summary_ClusterWidth_OnTrack@mean,402653184@Summary_ClusterWidth_OnTrack@mean,469762048@Summary_ClusterWidth_OnTrack@mean", "ClusterWidth_OnTrack.gif", 0, "369098752@Summary_ClusterWidth_OnTrack@entries > 10000 && 369098752@Summary_ClusterWidth_OnTrack@mean > 0", Start, End);

  //A.createTrend("369098752@Summary_ClusterStoN_OffTrack@mean,436207616@Summary_ClusterStoN_OffTrack@mean,402653184@Summary_ClusterStoN_OffTrack@mean,469762048@Summary_ClusterStoN_OffTrack@mean", "ClusterStoN_OffTrack.gif", 0, "369098752@Summary_ClusterStoN_OffTrack@entries > 10000 && 369098752@Summary_ClusterStoN_OffTrack@mean > 0", Start, End);
  //A.createTrend("369098752@Summary_TotalNumberOfDigis@mean,436207616@Summary_TotalNumberOfDigis@mean,402653184@Summary_TotalNumberOfDigis@mean,469762048@Summary_TotalNumberOfDigis@mean", "TotalNumberOfDigis.gif", 0, "369098752@Summary_TotalNumberOfDigis@entries > 10000 && 369098752@Summary_TotalNumberOfDigis@mean > 0", Start, End);


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
