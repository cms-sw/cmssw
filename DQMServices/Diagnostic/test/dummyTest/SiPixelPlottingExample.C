
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
  FWLiteEnabler::enable();




  HDQMInspectorConfigSiPixel PixelConfig;
  HDQMInspector A(&PixelConfig);
  A.setDB("sqlite_file:dbfile.db","HDQM_SiPixel","cms_cond_strip","w3807dev","");


  A.setDebug(1);
  A.setDoStat(1);


  A.createTrendLastRuns("1@SUMOFF_adc@yMean", "adc_yMean.gif", 0, "", NRuns);
  A.createTrendLastRuns("1@SUMOFF_charge_OffTrack@yMean", "charge_OffTrack_yMean.gif", 0, "", NRuns);

  A.closeFile();


  return;


}




