{
  gROOT->ProcessLine(".L fit2DProj.C+");
  // gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSPt\", \"0_MuScleFit.root\", \"1_MuScleFit.root\", \"Resonance mass vs pt\")" );
  gROOT->ProcessLine( "macroPlot(\"hRecBestResVSMu_MassVSEta\", \"0_MuScleFit.root\", \"1_MuScleFit.root\", \"Resonance mass vs #eta\")" );

//  gROOT->Reset();
}
