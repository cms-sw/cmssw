
{
/////////////////////////////////////////////////////////////////
//
// Extraction of the summary information using 
// DQMServices/Diagnostic/test/HDQMInspector.
// The sqlite database should have been filled using the new
// SiPixelHistoryDQMService.   
//
/////////////////////////////////////////////////////////////////


  int const NRuns = 2;

  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();




  HDQMInspectorConfigSiPixel PixelConfig;
  HDQMInspector A;
  A.setDB("sqlite_file:dbfile.db","HDQM_Pixel","cms_cond_strip","w3807dev","");


  A.setDebug(1);
  A.setDoStat(1);


  A.createTrendLastRuns("1@SUMOFF_ClustX@yMean", "1ClusterXsize_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "", NRuns);

  A.closeFile();


  return;


}




