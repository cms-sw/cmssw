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
    //TH2F* hist;
    //hist = (TH2F*)f.FindObjectAny("reportSummaryMap");
    TDirectoryFile* hist;	
    hist = (TDirectoryFile*)f.FindObjectAny("reportSummaryContents");
    if (0 == hist) 
    {
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
