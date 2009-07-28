#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiPixelHistoricInfoClient/test/HDQMInspectorConfigSiPixel.h"
#include <string>

void SiPixelHDQMInspector (const std::string & tagName, int const NRuns) {
/////////////////////////////////////////////////////////////////
//
// Extraction of the summary information using 
// DQMServices/Diagnostic/test/HDQMInspector.
// The sqlite database should have been filled using the new
// SiPixelHistoryDQMService.   
//
/////////////////////////////////////////////////////////////////


  //std::map<int, std::string> pixelTranslator = sipixelsummary::GetMap();

  //pixelTranslator Translator;

  //AutoLibraryLoader::enable();

  HDQMInspectorConfigSiPixel PixelConfig;
  //HDQMInspector A(&PixelConfig);
  HDQMInspector A(&PixelConfig);
  A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");

  A.createTrendLastRuns("1@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_charge_OffTrack@yMean", "charge_OffTrack_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_charge_OnTrack@yMean", "charge_OnTrack_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nclusters_OffTrack@yMean", "nclusters_OffTrack_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nclusters_OnTrack@yMean", "nclusters_OnTrack_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_size_OffTrack@yMean", "size_OffTrack_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_size_OnTrack@yMean", "size_OnTrack_yMean.gif", 0, "", NRuns);




  A.closeFile();


  return;


}


void SiPixelHDQMInspector (const std::string &tagName, int const Start, int const End) {
/////////////////////////////////////////////////////////////////
//
// Extraction of the summary information using 
// DQMServices/Diagnostic/test/HDQMInspector.
// The sqlite database should have been filled using the new
// SiPixelHistoryDQMService.   
//
/////////////////////////////////////////////////////////////////


  //std::map<int, std::string> pixelTranslator = sipixelsummary::GetMap();

  //pixelTranslator Translator;

  //AutoLibraryLoader::enable();

  HDQMInspectorConfigSiPixel PixelConfig;
  //HDQMInspector A(&PixelConfig);
  HDQMInspector A(&PixelConfig);
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
    SiPixelHDQMInspector( argv[1], atoi(argv[2]) );
  } else if(argc == 4) {
    std::cout << "Creating trends for range:  " << argv[2] << " " << argv[3] << " for tag: " << argv[1] << std::endl;
    SiPixelHDQMInspector( argv[1], atoi(argv[2]), atoi(argv[3]) );
  }

  return 0;
}
