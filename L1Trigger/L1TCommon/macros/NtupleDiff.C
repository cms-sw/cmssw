
#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"

void bitwise_compare(const char * tag, TTree * tree1, TTree * tree2, const char * var, const char * cut, int nbins, double max, double min){
  char command[1000];
  static int count = 0;

  TH1F h1("h1","",nbins,max,min);
  TH1F h2("h2","",nbins,max,min);
  sprintf(command, "%s>>h1", var);
  tree1->Draw(command, cut);
  sprintf(command, "%s>>h2", var);
  tree2->Draw(command, cut);
  //cout << "count 1:  " << h1.GetEntries() << "\n";
  //cout << "count 2:  " << h2.GetEntries() << "\n";
  
  if(!TString(var).CompareTo("sumEt[0]")) { var = "ETT";}
  else if(!TString(var).CompareTo("sumEt[1]")) {var = "HTT";}
  else if(!TString(var).CompareTo("sumEt[2]")) {var = "ETM";}
  else if(!TString(var).CompareTo("sumEt[3]")) {var = "HTM";}

  int fail = 0;
  for (int i=0; i<nbins+2; i++){
    if (h1.GetBinContent(i) != h2.GetBinContent(i)){
      fail = 1;
      cout << "discrepancy in " << var << " bin:  " << i << " " << h1.GetBinContent(i) << " vs " << h2.GetBinContent(i) << "\n";
    }
  }
  if (fail) {
    cout << "FAILURE:  variable " << var << " shows a disagreement\n";
    char name[1000];
    sprintf(name,"discrep_%s_%s_%d.pdf",tag,var,count);
    TCanvas c1;
    h1.GetXaxis()->SetTitle(var);
    h1.Draw("L");
    h2.Draw("epSAME");
    c1.SaveAs(name);
    count++;
  } else {
    cout << "SUCCESS:  bitwise equality for variable " << var << "\n";
  }
  
}




void NtupleDiff(const char * tag, const char * file1, const char * file2, const char * treepath1="l1UpgradeEmuTree/L1UpgradeTree", const char * treepath2="l1UpgradeEmuTree/L1UpgradeTree"){
  cout << "INFO: comparing contents of tree " << treepath1 << " in file " << file1 << "\n";
  cout << "INFO: comparing contents of tree " << treepath2 << " in file " << file2 << "\n";

  TFile input1(file1);
  TTree * tree1  = (TTree*) input1.Get(treepath1);
  if (! tree1) {
    cout << "ERROR: could not open tree 1.\n";
    input1.ls();
    return;
  }

  TFile input2(file2);
  TTree * tree2  = (TTree*) input2.Get(treepath2);
  if (! tree2) {
    cout << "ERROR: could not open tree 2.\n";
    input2.ls();
    return;
  }

  tree1->Print();
  tree2->Print();

  cout << "Very Central Jets:\n";
  bitwise_compare(tag, tree1, tree2, "jetEt", "(jetEt > 10.0) && (abs(jetEta) < 1.5)", 20.0, 0.0, 200.0);
  bitwise_compare(tag, tree1, tree2, "jetEta", "(jetEt > 10.0) && (abs(jetEta) < 1.5)", 20.0, -5.0, 5.0);
  bitwise_compare(tag, tree1, tree2, "jetPhi", "(jetEt > 10.0) && (abs(jetEta) < 1.5)", 20.0, -6.2, 6.2);

  cout << "All Jets:\n";
  bitwise_compare(tag, tree1, tree2, "jetEt", "(jetEt > 10.0)", 20.0, 0.0, 200.0);
  bitwise_compare(tag, tree1, tree2, "jetEta", "(jetEt > 10.0)", 20.0, -5.0, 5.0);
  bitwise_compare(tag, tree1, tree2, "jetPhi", "(jetEt > 10.0)", 20.0, -6.2, 6.2);

  bitwise_compare(tag, tree1, tree2, "tauEt", "tauEt > 10.0", 20.0, 0.0, 200.0);
  bitwise_compare(tag, tree1, tree2, "tauEta", "tauEt > 10.0", 20.0, -5.0, 5.0);
  bitwise_compare(tag, tree1, tree2, "tauPhi", "tauEt > 10.0", 20.0, -6.2, 6.2);

  bitwise_compare(tag, tree1, tree2, "egEt", "egEt > 10.0", 20.0, 0.0, 200.0);
  bitwise_compare(tag, tree1, tree2, "egEta", "egEt > 10.0", 20.0, -5.0, 5.0);
  bitwise_compare(tag, tree1, tree2, "egPhi", "egEt > 10.0", 20.0, -6.2, 6.2);

  bitwise_compare(tag, tree1, tree2, "muonEt", "muonEt > 10.0", 20.0, 0.0, 200.0);
  bitwise_compare(tag, tree1, tree2, "muonEta", "muonEt > 10.0", 20.0, -5.0, 5.0);
  bitwise_compare(tag, tree1, tree2, "muonPhi", "muonEt > 10.0", 20.0, -6.2, 6.2);

  bitwise_compare(tag, tree1, tree2, "sumEt[0]", "", 20.0, 0.0, 500.0);
  bitwise_compare(tag, tree1, tree2, "sumEt[1]", "", 20.0, 0.0, 500.0);
  bitwise_compare(tag, tree1, tree2, "sumEt[2]", "", 20.0, 0.0, 500.0);
  bitwise_compare(tag, tree1, tree2, "sumEt[3]", "", 20.0, 0.0, 500.0);
  

  //TH1F * jetEt = new TH1F("jetEt","", 20, 0.0, 200.0);
  //tree->Draw("jetEt>>jetEt","jetEt > 10.0");
  //cout << "jet count:  " << jetEt->GetEntries() << "\n";

  //TH1F * egEt = new TH1F("egEt","", 20, 0.0, 200.0);
  //tree->Draw("egEt>>egEt","egEt > 10.0");
  //cout << "eg count:  " << egEt->GetEntries() << "\n";

  //TH1F * tauEt = new TH1F("tauEt","", 20, 0.0, 200.0);
  //tree->Draw("tauEt>>tauEt","tauEt > 10.0");
  //cout << "tau count:  " << tauEt->GetEntries() << "\n";

  //TH1F * muonEt = new TH1F("muonEt","", 20, 0.0, 200.0);
  //tree->Draw("muonEt>>muonEt","muonEt > 1.0");
  //cout << "muon count:  " << muonEt->GetEntries() << "\n";  
}
