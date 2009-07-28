#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiPixelHistoricInfoClient/test/HDQMInspectorConfigSiPixel.h"

void SiPixelHDQMInspector (int const NRuns) {
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
  A.setDB("sqlite_file:dbfile.db","HDQM_SiPixel","cms_cond_strip","w3807dev","");


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


void SiPixelHDQMInspector (int const Start, int const End) {
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
  A.setDB("sqlite_file:dbfile.db","HDQM_SiPixel","cms_cond_strip","w3807dev","");


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
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [NRuns]" << std::endl;
    std::cerr << "Or:    " << argv[0] << " [FirstRun] [LastRun]" << std::endl;
    return 1;
  }

  if (argc == 2) {
    std::cout << "Creating trends for NRuns = " << argv[1] << std::endl;    
    SiPixelHDQMInspector( atoi(argv[1]) );
  } else if(argc == 3) {
    std::cout << "Creating trends for range:  " << argv[1] << " " << argv[2] << std::endl;    
    SiPixelHDQMInspector( atoi(argv[1]), atoi(argv[2]) );
  }

  return 0;
}
