void PlotAll() {
  char *username = getenv("USER");

  gROOT->ProcessLine(".L HistoManager.cc+");
  gROOT->ProcessLine(".L HcalVisualSelector.C+");
  gROOT->ProcessLine(".L HcalElectronicsSelector.C+");
  gROOT->ProcessLine(".L PlotAllDisplay.C+");
  gROOT->ProcessLine(".L PlotAllMenu.C+");
  // Popup the GUI...
  char s[128];
  sprintf (s, "new  PlotAllMenu(gClient->GetRoot(),200,200,\"%s\")",username);
  gROOT->ProcessLine(s);
}
