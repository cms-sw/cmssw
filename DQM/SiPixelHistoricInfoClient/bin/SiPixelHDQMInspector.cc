#include "DQMServices/Diagnostic/test/HDQMInspector.h"
#include "DQM/SiPixelHistoricInfoClient/interface/HDQMInspectorConfigSiPixel.h"

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

  A.createTrendLastRuns("1@SUMOFF_ClustX@ymean", "1ClusterXsize_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ClustX@entries", "1ClusterXsize_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ClustY@ymean", "1ClusterYsize_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ClustY@entries", "1ClusterYsize_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_adc@ymean", "adc_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_adc@entries", "adc_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_charge@ymean", "charge_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_charge@entries", "charge_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nRecHits@ymean", "nRecHits_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nRecHits@entries", "nRecHits_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nclusters@ymean", "nclusters_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_nclusters@entries", "nclusters_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ndigis@ymean", "ndigis_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_ndigis@entries", "ndigis_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_sizeX@ymean", "sizeX_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_sizeX@entries", "sizeX_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_sizeY@ymean", "sizeY_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_sizeY@entries", "sizeY_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_x@ymean", "x_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_x@entries", "x_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_y@ymean", "y_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_y@entries", "y_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_maxrow@ymean", "maxrow_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_maxrow@entries", "maxrow_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_minrow@ymean", "minrow_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_minrow@entries", "minrow_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_maxcol@ymean", "maxcol_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_maxcol@entries", "maxcol_entries.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_mincol@ymean", "mincol_ymean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_mincol@entries", "mincol_entries.gif", 0, "", NRuns);

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
