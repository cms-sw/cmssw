#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TDirectory.h>
#include <TH1F.h>

#include <iostream>

TString gTreePath = "/data1/emanuele/monox/TREES_060515_MET200SKIM/%s/treeProducerDarkMatterMonoJet/tree.root";
TString gFriendTreePath = "/data1/emanuele/monox/TREES_060515_MET200SKIM/0_eventvars_mj_v1/evVarFriend_%s.root";

void fillEF(TString hist, TString presel, TString srcut, TString crcut, TString weight, TString compName) {
  TDirectory *root = gDirectory;
  TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
  TTree *tree = (TTree*) f->Get("tree");
  tree->AddFriend("mjvars/t",Form(gFriendTreePath.Data(),compName.Data()));

  root->cd();
  TString mycut = presel;
  TString mydencut = Form("((%s) && (%s))", mycut.Data(), crcut.Data());
  TString mynumcut = Form("((%s) && (%s))", mycut.Data(), srcut.Data());

  tree->Draw(Form("min(metNoMu_pt,1000)>>+%s_den", hist.Data()), Form("%s * %s", mydencut.Data(), weight.Data()));
  tree->Draw(Form("min(metNoMu_pt,1000)>>+%s_num", hist.Data()), Form("%s * %s", mynumcut.Data(), weight.Data()));

  f->Close();

}

void fillOneComp(TString hist, TString presel, TString rcut, TString weight, TString compName) {
  TDirectory *root = gDirectory;
  TFile *f = TFile::Open(Form(gTreePath.Data(),compName.Data()));
  TTree *tree = (TTree*) f->Get("tree");
  tree->AddFriend("mjvars/t",Form(gFriendTreePath.Data(),compName.Data()));

  root->cd();
  TString mycut = presel;
  TString myfinalcut = Form("((%s) && (%s))", mycut.Data(), rcut.Data());

  tree->Draw(Form("min(metNoMu_pt,1000)>>+%s", hist.Data()), Form("%s * %s", myfinalcut.Data(), weight.Data()));

  f->Close();

}

void fillWTrivialExtrapolationFactorsFromMC() {
  const int nmet = 9;
  double metbins[nmet+1] = { 200, 225, 250, 275, 300, 325, 350, 400, 500, 1000 };

  TFile *fOut = TFile::Open("ef_W.root", "RECREATE");
 
  TH1F *EF_W_den = new TH1F("EF_W_den","",nmet,metbins);
  TH1F *EF_W_num = new TH1F("EF_W_num","",nmet,metbins);

  TString baseCut = " ((nJet30 == 1 || nJet30 == 2) && jetclean > 0.5) && (Jet[0]_pt>110) && (nJet30 == 1 || (nJet==2 && abs(dphijj)<2.5)) && (nGamma15V==0) && (nEle10V==0 && nTau15V==0) ";
  
  const char *samples[4] = { "WJetsToLNu_HT100to200", "WJetsToLNu_HT200to400", "WJetsToLNu_HT400to600", "WJetsToLNu_HT600toInf" };
 
  TString hist = "EF_W";
 
  for (int id = 0; id < 4; ++id) { 
    TString sample = TString(samples[id]);
    std::cout << "Processing Control Region and Signal Region selection on " << sample << std::endl;
    fillEF(hist, baseCut, "nMu10V == 0", "nMu10V > 0 && nMu20T == 1 && nBJetMedium30 == 0", "weight", sample);
  }

  TH1 *den = (TH1*) EF_W_den; den->Sumw2(); den->Write();
  TH1 *num = (TH1*) EF_W_num; num->Sumw2(); num->Write();

  TH1 *ratio = (TH1*) num->Clone(hist);
  ratio->Divide(num,den,1,1);
  ratio->Write();

  fOut->Close();

}

void fillZTrivialExtrapolationFactorsFromMC() {
  const int nmet = 9;
  double metbins[nmet+1] = { 200, 225, 250, 275, 300, 325, 350, 400, 500, 1000 };

  TFile *fOut = TFile::Open("ef_Z.root", "RECREATE");
 
  TH1F *EF_Z_den = new TH1F("EF_Z_den","",nmet,metbins);
  TH1F *EF_Z_num = new TH1F("EF_Z_num","",nmet,metbins);

  TString baseCut = " ((nJet30 == 1 || nJet30 == 2) && jetclean > 0.5) && (Jet[0]_pt>110) && (nJet30 == 1 || (nJet==2 && abs(dphijj)<2.5)) && (nGamma15V==0) && (nEle10V==0 && nTau15V==0) ";
  
  const char *samples_CR[4] = { "DYJetsToLL_M50_HT100to200", "DYJetsToLL_M50_HT200to400", "DYJetsToLL_M50_HT400to600", "DYJetsToLL_M50_HT600toInf" };
  const char *samples_SR[4] = { "ZJetsToNuNu_HT100to200", "ZJetsToNuNu_HT200to400", "ZJetsToNuNu_HT400to600", "ZJetsToNuNu_HT600toInf" };
 
  TString hist = "EF_Z";
 
  for (int id = 0; id < 4; ++id) { 
    TString sample = TString(samples_CR[id]);
    std::cout << "Processing Control Region selection on " << sample << std::endl;
    fillOneComp(hist+"_den", baseCut, "(nMu10V == 2 && LepGood[0]_pdgId == -LepGood[1]_pdgId) && (abs(LepGood[0]_pdgId) == 13) && (LepGood[0]_tightId > 0 && LepGood[0]_relIso04 < 0.12) && (mZ1 > 60 && mZ1 < 120)", "weight", sample);
  }

  for (int id = 0; id < 4; ++id) { 
    TString sample = TString(samples_SR[id]);
    std::cout << "Processing Signal Region selection on " << sample << std::endl;
    fillOneComp(hist+"_num", baseCut, "nMu10V == 0", "weight", sample);
  }

  TH1 *den = (TH1*) EF_Z_den; den->Sumw2(); den->Write();
  TH1 *num = (TH1*) EF_Z_num; num->Sumw2(); num->Write();

  TH1 *ratio = (TH1*) num->Clone(hist);
  ratio->Divide(num,den,1,1);
  ratio->Write();

  fOut->Close();

}

