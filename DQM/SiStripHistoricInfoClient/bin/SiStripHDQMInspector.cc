#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiStripHistoricInfoClient/test/HDQMInspectorConfigSiStrip.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  HDQMInspector A(&StripConfig);
  //HDQMInspector A;
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_COND_STRIP",tagName,"cms_cond_strip","WCYE6II08K530GPK","");


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");

  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OnTrack@mean,436207616@Summary_TotalNumberOfClusters_OnTrack@mean,402653184@Summary_TotalNumberOfClusters_OnTrack@mean,469762048@Summary_TotalNumberOfClusters_OnTrack@mean", "OnTrackClusters.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_TotalNumberOfClusters_OffTrack@mean,436207616@Summary_TotalNumberOfClusters_OffTrack@mean,402653184@Summary_TotalNumberOfClusters_OffTrack@mean,469762048@Summary_TotalNumberOfClusters_OffTrack@mean", "TotalNumberOfClusters_OffTrack.gif", 0, "", NRuns);





  A.createTrendLastRuns("369098752@Summary_ClusterChargeCorr_OnTrack@userExample_XMax,436207616@Summary_ClusterChargeCorr_OnTrack@userExample_XMax,402653184@Summary_ClusterChargeCorr_OnTrack@userExample_XMax,469762048@Summary_ClusterChargeCorr_OnTrack@userExample_XMax", "ClusterChargeCorr_OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterCharge_OffTrack@userExample_XMax,436207616@Summary_ClusterCharge_OffTrack@userExample_XMax,402653184@Summary_ClusterCharge_OffTrack@userExample_XMax,469762048@Summary_ClusterCharge_OffTrack@userExample_XMax", "ClusterCharge_OffTrack.gif", 0, "", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterNoise_OnTrack@userExample_XMax,436207616@Summary_ClusterNoise_OnTrack@userExample_XMax,402653184@Summary_ClusterNoise_OnTrack@userExample_XMax,469762048@Summary_ClusterNoise_OnTrack@userExample_XMax", "ClusterNoise_OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise_OffTrack@userExample_XMax,436207616@Summary_ClusterNoise_OffTrack@userExample_XMax,402653184@Summary_ClusterNoise_OffTrack@userExample_XMax,469762048@Summary_ClusterNoise_OffTrack@userExample_XMax", "ClusterNoise_OffTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoNCorr_OnTrack@userExample_XMax,436207616@Summary_ClusterStoNCorr_OnTrack@userExample_XMax,402653184@Summary_ClusterStoNCorr_OnTrack@userExample_XMax,469762048@Summary_ClusterStoNCorr_OnTrack@userExample_XMax", "ClusterStoNCorr_OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoN_OffTrack@userExample_XMax,436207616@Summary_ClusterStoN_OffTrack@userExample_XMax,402653184@Summary_ClusterStoN_OffTrack@userExample_XMax,469762048@Summary_ClusterStoN_OffTrack@userExample_XMax", "ClusterStoN_OffTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth_OnTrack@mean,436207616@Summary_ClusterWidth_OnTrack@mean,402653184@Summary_ClusterWidth_OnTrack@mean,469762048@Summary_ClusterWidth_OnTrack@mean", "ClusterWidth_OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth_OffTrack@mean,436207616@Summary_ClusterWidth_OffTrack@mean,402653184@Summary_ClusterWidth_OffTrack@mean,469762048@Summary_ClusterWidth_OffTrack@mean", "ClusterWidth_OffTrack.gif", 0, "", NRuns);


  A.createTrendLastRuns("369098752@Summary_TotalNumberOfDigis@userExample_XMax,436207616@Summary_TotalNumberOfDigis@userExample_XMax,402653184@Summary_TotalNumberOfDigis@userExample_XMax,469762048@Summary_TotalNumberOfDigis@userExample_XMax", "TotalNumberOfDigis.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterChargeCorr__OnTrack@userExample_XMax,436207616@Summary_ClusterChargeCorr__OnTrack@userExample_XMax,402653184@Summary_ClusterChargeCorr__OnTrack@userExample_XMax,469762048@Summary_ClusterChargeCorr__OnTrack@userExample_XMax", "_ClusterChargeCorr__OnTrack.gif", 0, "", NRuns);

  A.createTrendLastRuns("369098752@Summary_ClusterCharge__OffTrack@userExample_XMax,436207616@Summary_ClusterCharge__OffTrack@userExample_XMax,402653184@Summary_ClusterCharge__OffTrack@userExample_XMax,469762048@Summary_ClusterCharge__OffTrack@userExample_XMax", "ClusterCharge__OffTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise__OnTrack@userExample_XMax,436207616@Summary_ClusterNoise__OnTrack@userExample_XMax,402653184@Summary_ClusterNoise__OnTrack@userExample_XMax,469762048@Summary_ClusterNoise__OnTrack@userExample_XMax", "ClusterNoise__OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterNoise__OffTrack@userExample_XMax,436207616@Summary_ClusterNoise__OffTrack@userExample_XMax,402653184@Summary_ClusterNoise__OffTrack@userExample_XMax,469762048@Summary_ClusterNoise__OffTrack@userExample_XMax", "ClusterNoise__OffTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoNCorr__OnTrack@userExample_XMax,436207616@Summary_ClusterStoNCorr__OnTrack@userExample_XMax,402653184@Summary_ClusterStoNCorr__OnTrack@userExample_XMax,469762048@Summary_ClusterStoNCorr__OnTrack@userExample_XMax", "ClusterStoNCorr__OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterStoN__OffTrack@userExample_XMax,436207616@Summary_ClusterStoN__OffTrack@userExample_XMax,402653184@Summary_ClusterStoN__OffTrack@userExample_XMax,469762048@Summary_ClusterStoN__OffTrack@userExample_XMax", "ClusterStoN__OffTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth__OnTrack@mean,436207616@Summary_ClusterWidth__OnTrack@mean,402653184@Summary_ClusterWidth__OnTrack@mean,469762048@Summary_ClusterWidth__OnTrack@mean", "ClusterWidth__OnTrack.gif", 0, "", NRuns);
  A.createTrendLastRuns("369098752@Summary_ClusterWidth__OffTrack@mean,436207616@Summary_ClusterWidth__OffTrack@mean,402653184@Summary_ClusterWidth__OffTrack@mean,469762048@Summary_ClusterWidth__OffTrack@mean", "ClusterWidth__OffTrack.gif", 0, "", NRuns);
  //A.createTrendLastRuns("369098752@XXX@mean,436207616@XXX@mean,402653184@XXX@mean,469762048@XXX@mean", "OnTrackClusters.gif", 0, "", NRuns);

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
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_COND_STRIP",tagName,"cms_cond_strip","WCYE6II08K530GPK","");


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");

  A.createTrend("369098752@Summary_ClusterChargeCorr__OnTrack@mean,436207616@Summary_ClusterChargeCorr__OnTrack@mean,402653184@Summary_ClusterChargeCorr__OnTrack@mean,469762048@Summary_ClusterChargeCorr__OnTrack@mean", "_ClusterChargeCorr__OnTrack.gif", 0, "", Start, End);


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
