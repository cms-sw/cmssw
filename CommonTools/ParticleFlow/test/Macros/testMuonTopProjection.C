{

  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  gSystem->Load("libCintex.so");
  ROOT::Cintex::Cintex::Enable();

  TFile fDisabled("tpDisabled.root");
  TTree *tDisabled = (TTree*) fDisabled.Get("Events");
  TFile fEnabled("tpEnabled.root");
  TTree *tEnabled = (TTree*) fEnabled.Get("Events");
  
  
  tDisabled.Draw("patJets_selectedPatJetsPFlow__PAT.obj.pt()>>h1");
  tEnabled.Draw("patJets_selectedPatJetsPFlow__PAT.obj.pt()>>h2","","same");
 
  h1.SetStats( false );
  h1.Draw();

  h2.SetLineColor(4);
  h2.Draw("same");

  TLegend leg(0.6,0.6,0.8,0.8);
  leg.AddEntry( &h1, "Muon TP disabled");
  leg.AddEntry( &h2, "Muon TP enabled");
  leg.Draw();
}
