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

  A.createTrendLastRuns("1@SUMOFF_ClustX@yMean", "1ClusterXsize_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ClustY@yMean", "1ClusterYsize_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_charge@yMean", "charge_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nclusters@yMean", "nclusters_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_sizeX@yMean", "sizeX_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_sizeY@yMean", "sizeY_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_x@yMean", "x_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_y@yMean", "y_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_maxrow@yMean", "maxrow_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_minrow@yMean", "minrow_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_maxcol@yMean", "maxcol_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_mincol@yMean", "mincol_yMean.gif", 0, "", NRuns);

  A.createTrendLastRuns("2@SUMOFF_ClustX@yMean", "1ClusterXsize_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_ClustY@yMean", "1ClusterYsize_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_charge@yMean", "charge_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_nRecHits@yMean", "nRecHits_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_nclusters@yMean", "nclusters_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_ndigis@yMean", "ndigis_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_sizeX@yMean", "sizeX_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_sizeY@yMean", "sizeY_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_x@yMean", "x_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_y@yMean", "y_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_maxrow@yMean", "maxrow_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_minrow@yMean", "minrow_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_maxcol@yMean", "maxcol_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("2@SUMOFF_mincol@yMean", "mincol_yMean.gif", 0, "", NRuns);

  A.closeFile();


  return;


}




int main (int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " [NRuns]" << std::endl;
    return 1;
  }

  SiPixelHDQMInspector( atoi(argv[1]) );

  return 0;
}
