{
  TFile *mctester1 = TFile::Open("prod1/mc-tester.root");
  TFile *mctester2 = TFile::Open("prod2/mc-tester.root");
  TFile *mcresults = TFile::Open("mc-results.root");
  if(!mctester1) printf("WARNING: cannot open prod1/mc-tester.root!\n");
  if(!mctester2) printf("WARNING: cannot open prod2/mc-tester.root!\n");
  if(!mcresults) printf("WARNING: cannot open ./mc-results.root!\n");
  TH1::AddDirectory(kFALSE);  
  //    TBrowser b("mc-tester",mcresults,"MC-TESTER results browser" );
  TBrowser b;
}
