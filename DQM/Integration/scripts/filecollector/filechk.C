int filechk(string fname)
{
  //TFile f("DQM_V0001_SiStrip_R000062940.root");
  TFile f(fname.c_str());

  if (f.IsZombie()){
    //cout << "File corrupted" << endl;
  return -1;
  }
  else
  {
    if ( fname.find("_SiStrip_") != string::npos ){
      TString rnStr = fname.substr(fname.find("_R") + 2, 9);
      TString runDirStr("Run ");
      runDirStr += rnStr.Atoi();
      TDirectoryFile* runDir = (TDirectoryFile*)f.FindObjectAny(runDirStr);
      if ( runDir == 0 )
        return 0; 
      
      TDirectoryFile* tracking = (TDirectoryFile*)runDir->FindObjectAny("Tracking");
      if ( tracking == 0 )
        return 0;
      
      TDirectoryFile* sistrip = (TDirectoryFile*)runDir->FindObjectAny("SiStrip");
      if ( sistrip == 0 )
        return 0;  
       
      TDirectoryFile* hist = (TDirectoryFile*)tracking->FindObjectAny("reportSummaryMap");
      if ( hist == 0 )
        return 0;
       
      hist = (TDirectoryFile*)sistrip->FindObjectAny("reportSummaryMap");
      if ( hist == 0 )
        return 0; 
      
      return 1;
    }
    
    //TH2F* hist;
    //hist = (TH2F*)f.FindObjectAny("reportSummaryMap");
    TDirectoryFile* hist;	
    //hist = (TDirectoryFile*)f.FindObjectAny("reportSummaryContents");
    hist = (TDirectoryFile*)f.FindObjectAny("reportSummaryMap");
    if (0 == hist) 
    {
      TDirectoryFile* hist2;
      hist2 = (TDirectoryFile*)f.FindObjectAny("EventInfo");
      if (0 != hist2)
        return 1;

      //cout << "File is incomplete" << endl;
      return 0;
    }
    else
    {
      //cout << "File is OK" << endl;
      return 1;
    }
  }
}
