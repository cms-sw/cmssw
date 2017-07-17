
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"

int CheckL1Ntuple(const char * file= "L1Ntuple.root", const char * treepath = "l1UpgradeEmuTree/L1UpgradeTree"){
  cout << "INFO: summary of content in file " << file << "\n";
  TFile input(file);
  TTree * tree  = (TTree*) input.Get(treepath);
  if (! tree) {
    cout << "ERROR: could not open tree.\n";
    input.ls();
    return 1;
  }
  tree->Print();

  TH1F * jetEt = new TH1F("jetEt","", 20, 0.0, 200.0);
  tree->Draw("jetEt>>jetEt","jetEt > 10.0");
  int jet_count = jetEt->GetEntries();

  TH1F * egEt = new TH1F("egEt","", 20, 0.0, 200.0);
  tree->Draw("egEt>>egEt","egEt > 10.0");
  int eg_count = egEt->GetEntries();

  TH1F * tauEt = new TH1F("tauEt","", 20, 0.0, 200.0);
  tree->Draw("tauEt>>tauEt","tauEt > 10.0");
  int tau_count = tauEt->GetEntries();

  TH1F * muonEt = new TH1F("muonEt","", 20, 0.0, 200.0);
  tree->Draw("muonEt>>muonEt","muonEt > 1.0");
  int muon_count = muonEt->GetEntries();


  int success = 1;
  if (jet_count <= 0)  success = 0;
  if (eg_count <= 0)   success = 0;
  if (tau_count <= 0)  success = 0;
  if (muon_count <= 0) success = 0;
  
  cout << "jet count:   " << jet_count << "\n";  
  cout << "eg count:    " << eg_count << "\n";  
  cout << "tau count:   " << tau_count << "\n";  
  cout << "muon count:  " << muon_count << "\n";  

  if (success){
    cout << "STATUS:  SUCCESS\n";
    return 0;
  } else {
    cout << "STATUS:  FAILURE\n";
    return 1;
  }
}
