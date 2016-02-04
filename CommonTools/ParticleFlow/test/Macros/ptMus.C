{


  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  gSystem->Load("libCintex.so");
  ROOT::Cintex::Cintex::Enable();
  TFile f("patTuple_PF2PAT.root");
  
  Events.Draw("patMuons_selectedPatMuons__PAT.obj.pt()>>h1");
  Events.Draw("patMuons_selectedPatMuonsPFlow__PAT.obj.pt()>>h2","","same");
 
  h1.SetStats( false );
  h1.Draw();

  h2.SetLineColor(4);
  h2.Draw("same");

  TLegend leg(0.6,0.6,0.8,0.8);
  leg.AddEntry( &h1, "std PAT");
  leg.AddEntry( &h2, "PF2PAT+PAT");
  leg.Draw();
}
