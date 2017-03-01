
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"

void NtupleSummary(const char * file, const char * treepath = "l1UpgradeEmuTree/L1UpgradeTree"){
  cout << "INFO: summary of content in file " << file << "\n";
  TFile input(file);
  TTree * tree  = (TTree*) input.Get(treepath);
  if (! tree) {
    cout << "ERROR: could not open tree.\n";
    input.ls();
    return;
  }
  tree->Print();

  TH1F * fjetEt = new TH1F("fjetEt","", 20, 0.0, 200.0);
  tree->Draw("jetEt>>fjetEt","(jetEt > 10.0) && (abs(jetEta) > 3.0)");
  cout << "forward jet count:  " << fjetEt->GetEntries() << "\n";

  TH1F * jetEt = new TH1F("jetEt","", 20, 0.0, 200.0);
  tree->Draw("jetEt>>jetEt","jetEt > 10.0");
  cout << "jet count:  " << jetEt->GetEntries() << "\n";

  TH1F * egEt = new TH1F("egEt","", 20, 0.0, 200.0);
  tree->Draw("egEt>>egEt","egEt > 10.0");
  cout << "eg count:  " << egEt->GetEntries() << "\n";

  TH1F * tauEt = new TH1F("tauEt","", 20, 0.0, 200.0);
  tree->Draw("tauEt>>tauEt","tauEt > 10.0");
  cout << "tau count:  " << tauEt->GetEntries() << "\n";

  TH1F * muonEt = new TH1F("muonEt","", 20, 0.0, 200.0);
  tree->Draw("muonEt>>muonEt","muonEt > 1.0");
  cout << "muon count:  " << muonEt->GetEntries() << "\n";  

  




}
