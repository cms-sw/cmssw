#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiPixelHistoricInfoClient/test/HDQMInspectorConfigSiPixel.h"
#include <string>

void SiPixelHDQMInspector (const std::string & tagName, std::string const& Password, int const NRuns) {
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
  HDQMInspector A(&PixelConfig);
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");



  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");

  A.createTrendLastRuns("0@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "0@SUMOFF_adc@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_charge_OffTrack@yMean", "charge_OffTrack_yMean.gif", 0, "0@SUMOFF_charge_OffTrack@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_charge_OnTrack@yMean", "charge_OnTrack_yMean.gif", 0, "0@SUMOFF_charge_OnTrack@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, "0@SUMOFF_nRecHits@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_nclusters_OffTrack@yMean", "nclusters_OffTrack_yMean.gif", 0, "0@SUMOFF_nclusters_OffTrack@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_nclusters_OnTrack@yMean", "nclusters_OnTrack_yMean.gif", 0, "0@SUMOFF_nclusters_OnTrack@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, "0@SUMOFF_ndigis@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_size_OffTrack@yMean", "size_OffTrack_yMean.gif", 0, "0@SUMOFF_size_OffTrack@yMean > 0", NRuns);
  A.createTrendLastRuns("0@SUMOFF_size_OnTrack@yMean", "size_OnTrack_yMean.gif", 0, "0@SUMOFF_size_OnTrack@yMean > 0", NRuns);
  A.createTrendLastRuns("0@ntracks_rsWithMaterialTracksP5@NTracksPixOverAll", "NTracksPixOverAll.gif", 0, "0@ntracks_rsWithMaterialTracksP5@NTracksPixOverAll > 0", NRuns);
  A.createTrendLastRuns("0@ntracks_rsWithMaterialTracksP5@NTracksFPixOverBPix", "NTracksFPixOverBPix.gif", 0, "0@ntracks_rsWithMaterialTracksP5@NTracksFPixOverBPix > 0", NRuns);




  A.closeFile();


  return;


}


void SiPixelHDQMInspector (const std::string &tagName, std::string const& Password, int const Start, int const End) {
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
  //A.setDB("sqlite_file:dbfile.db",tagName,"cms_cond_strip","w3807dev","");
  A.setDB("oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE",tagName,"cms_dqm_31x_offline", Password,"");


  A.setDebug(1);
  A.setDoStat(1);

  //A.setBlackList("68286");
  A.createTrend("0@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "0@SUMOFF_adc@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_charge_OffTrack@yMean", "charge_OffTrack_yMean.gif", 0, "0@SUMOFF_charge_OffTrack@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_charge_OnTrack@yMean", "charge_OnTrack_yMean.gif", 0, "0@SUMOFF_charge_OnTrack@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, "0@SUMOFF_nRecHits@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_nclusters_OffTrack@yMean", "nclusters_OffTrack_yMean.gif", 0, "0@SUMOFF_nclusters_OffTrack@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_nclusters_OnTrack@yMean", "nclusters_OnTrack_yMean.gif", 0, "0@SUMOFF_nclusters_OnTrack@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, "0@SUMOFF_ndigis@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_size_OffTrack@yMean", "size_OffTrack_yMean.gif", 0, "0@SUMOFF_size_OffTrack@yMean > 0", Start, End);
  A.createTrend("0@SUMOFF_size_OnTrack@yMean", "size_OnTrack_yMean.gif", 0, "0@SUMOFF_size_OnTrack@yMean > 0", Start, End);
  A.createTrend("0@ntracks_rsWithMaterialTracksP5@NTracksPixOverAll", "NTracksPixOverAll.gif", 0, "0@ntracks_rsWithMaterialTracksP5@NTracksPixOverAll > 0", Start, End);
  A.createTrend("0@ntracks_rsWithMaterialTracksP5@NTracksFPixOverBPix", "NTracksFPixOverBPix.gif", 0, "0@ntracks_rsWithMaterialTracksP5@NTracksFPixOverBPix > 0", Start, End);

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
    SiPixelHDQMInspector( argv[1], argv[2], atoi(argv[3]) );
  } else if(argc == 5) {
    std::cout << "Creating trends for range:  " << argv[3] << " " << argv[4] << " for tag: " << argv[1] << std::endl;
    SiPixelHDQMInspector( argv[1], argv[2], atoi(argv[3]), atoi(argv[4]) );
  }

  return 0;
}
