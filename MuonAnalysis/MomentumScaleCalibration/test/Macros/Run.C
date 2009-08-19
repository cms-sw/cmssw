{
  gROOT->ProcessLine(".L fit2DProj.C+");
  // macroPlot parameters are: histogram name, input file 1 name, input file 2 name, output histogram title, resonance type,
  // rebinX, rebinY, fitType (1:gaussian; 2:lorentzian, ...)

  // Y: gaussian fit
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPt\", \"0_MuScleFit_Y.root\", \"1_MuScleFit_Y.root\", \"Resonance mass vs pt\", \"Y\", 4, 4, 1)" );
  gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSEta\", \"0_MuScleFit_Y.root\", \"1_MuScleFit_Y.root\", \"Resonance mass vs #eta\", \"Y\", 4, 4, 1)" );

  // Z: lorentzian fit
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPt\", \"0_MuScleFit_Y.root\", \"1_MuScleFit_Y.root\", \"Resonance mass vs pt\", \"Z\", 4, 4, 2)" );
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSEta\", \"0_MuScleFit_Y.root\", \"1_MuScleFit_Y.root\", \"Resonance mass vs #eta\", \"Z\", 4, 4, 2)" );

  // gROOT->Reset();
}
