{

  gSystem->Load("libFWCoreFWLite.so");
  FWLiteEnabler::enable();

  TFile fDisabled("tpDisabled.root");
  TTree *tDisabled = (TTree*) fDisabled.Get("Events");
  TFile fEnabled("tpEnabled.root");
  TTree *tEnabled = (TTree*) fEnabled.Get("Events");
  
  
  if(tDisabled != 0) tDisabled->Draw("patJets_selectedPatJetsPFlow__PAT.obj.pt()>>h1");
  if(tEnabled != 0) tEnabled->Draw("patJets_selectedPatJetsPFlow__PAT.obj.pt()>>h2","","same");
 
  TH1* h1 = 0;
  gDirectory->GetObject("h1", h1);
  h1->SetStats( false );
  h1->Draw();

  TH1* h2 = 0;
  gDirectory->GetObject("h2", h2);
  h2->SetLineColor(4);
  h2->Draw("same");

  TLegend leg(0.6,0.6,0.8,0.8);
  leg.AddEntry( h1, "Muon TP disabled");
  leg.AddEntry( h2, "Muon TP enabled");
  leg.Draw();
}
