{
  gROOT->LoadMacro("macros.C");     // Load service macros
  gSystem->Load("../../../../lib/slc3_ia32_gcc323/libtestDTCalibration.so");

  // Get the style
  TStyle * style = getStyle();
  // Style options
  style->SetOptStat(111111);
  style->SetOptFit(1111);
  style->cd();

  // Retrieve histogram sets
  TFile *f = gROOT->GetListOfFiles()->Last();
  DTTimeBoxPlotter *plotter = new DTTimeBoxPlotter(f);


}

