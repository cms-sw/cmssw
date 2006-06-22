{

  gSystem->Load("libFWCoreFWLite.so"); 
  AutoLibraryLoader::enable();
  TChain chain("Events");
  chain.Add("evtgen_jets.root");
  chain.Add("evtgen_jets2.root");
  chain.GetEntry(0);  // This works and is pretty easy
  //TFile file("evtgen_jets.root");  //This works, but is too difficult.
  //gROOT->GetClass("CaloJet");  //This works but only for CaloJet
  //gROOT->GetClass("vector<CaloJet>");  //This does NOT work.
}
