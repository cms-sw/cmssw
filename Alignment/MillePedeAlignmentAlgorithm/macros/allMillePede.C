// Original Author: Gero Flucke
// last change    : $Date: 2012/05/12 19:20:58 $
// by             : $Author: flucke $

void allMillePede(bool forceCompile = false) {
  TString compile(forceCompile ? "++" : "+");
  gROOT->ProcessLine(".L GFUtils/GFHistArray.C" + compile);
  gROOT->ProcessLine(".L GFUtils/GFHistManager.C" + compile);

  gROOT->ProcessLine(".L MillePedeTrees.C" + compile);
  gROOT->ProcessLine(".L PlotMillePede.C" + compile);
  gROOT->ProcessLine(".L PlotMillePedeIOV.C" + compile);
  gROOT->ProcessLine(".L CompareMillePede.C" + compile);

  gROOT->ProcessLine(".L PlotMilleMonitor.C" + compile);
}



