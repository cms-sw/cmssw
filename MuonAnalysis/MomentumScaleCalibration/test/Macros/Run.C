{
  gROOT->ProcessLine(".L fit2DProj.C+");
  // macroPlot parameters are: histogram name, input file 1 name, input file 2 name, output histogram title, resonance type,
  // rebinX, rebinY, fitType (1:gaussian; 2:lorentzian, ...), output file name (default "filegraph.root")

  // Y: gaussian fit
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPt\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs pt\", \"Y\", 4, 4, 1, \"filegraph_pt.root\")" );
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSEta\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs #eta\", \"Y\", 2, 2, 1, \"filegraph_eta.root\")" );
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPhiPlus\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs #phi\", \"Y\", 2, 2, 1, \"filegraph_phi.root\")" );

  // Z: lorentzian fit
  gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPt\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs pt\", \"Z\", 4, 4, 2, \"filegraph_pt.root\")" );
  gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSEta\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs #eta\", \"Z\", 4, 4, 2, \"filegraph_eta.root\")" );
  gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPhiPlus\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs #phi\", \"Z\", 4, 4, 2, \"filegraph_phi.root\")" );

  // gROOT->Reset();
}
