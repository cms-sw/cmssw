#ifndef __L1Analysis_L1AnalysisL1Extra_H__
#define __L1Analysis_L1AnalysisL1Extra_H__

#include <TTree.h>
#include <TMatrixD.h>

namespace L1Analysis
{
  class L1AnalysisL1Extra
{

  public : 
  void initTree(TTree * tree, const std::string & className);

  public:
  L1AnalysisL1Extra() {}
  void print(); 
  bool check();   
  
    // ---- L1AnalysisL1Extra information.
    unsigned nIsoEm;
    std::vector<double> isoEmEt;
    std::vector<double> isoEmEta;
    std::vector<double> isoEmPhi;
    std::vector<int>    isoEmBx;

    unsigned nNonIsoEm;
    std::vector<double> nonIsoEmEt;
    std::vector<double> nonIsoEmEta;
    std::vector<double> nonIsoEmPhi;
    std::vector<int>    nonIsoEmBx;

    unsigned nCenJets;
    std::vector<double> cenJetEt;
    std::vector<double> cenJetEta;
    std::vector<double> cenJetPhi; 
    std::vector<int>    cenJetBx;

    unsigned nFwdJets;
    std::vector<double> fwdJetEt;
    std::vector<double> fwdJetEta;
    std::vector<double> fwdJetPhi;
    std::vector<int>    fwdJetBx;

    unsigned nTauJets;
    std::vector<double> tauJetEt;
    std::vector<double> tauJetEta;
    std::vector<double> tauJetPhi;
    std::vector<int>    tauJetBx;
    
    unsigned nMuons;
    std::vector<double>   muonEt;
    std::vector<double>   muonEta;
    std::vector<double>   muonPhi;
    std::vector<int>      muonChg;
    std::vector<unsigned int> muonIso;
    std::vector<unsigned int> muonFwd;
    std::vector<unsigned int> muonMip;
    std::vector<unsigned int> muonRPC;
    std::vector<int>          muonBx;

    std::vector<double> hfEtSum;
    std::vector<unsigned int> hfBitCnt; 
    std::vector<int>          hfBx;
 
    double met;
    double metPhi;
    int metsBx;
    double mht;
    double mhtPhi;
    int mhtsBx;
    double et;
    double ht;
};
}

#endif

#ifdef l1ntuple_cxx



void L1Analysis::L1AnalysisL1Extra::initTree(TTree * tree, const std::string & className)
{
   SetBranchAddress(tree, "nIsoEm", className,  &nIsoEm);
   SetBranchAddress(tree, "isoEmEt", className,  &isoEmEt);
   SetBranchAddress(tree, "isoEmEta", className,  &isoEmEta);
   SetBranchAddress(tree, "isoEmPhi", className,  &isoEmPhi); 
   SetBranchAddress(tree, "isoEmBx", className,  &isoEmBx);
   SetBranchAddress(tree, "nNonIsoEm", className,  &nNonIsoEm);
   SetBranchAddress(tree, "nonIsoEmEt", className,  &nonIsoEmEt);
   SetBranchAddress(tree, "nonIsoEmEta", className,  &nonIsoEmEta);
   SetBranchAddress(tree, "nonIsoEmPhi", className,  &nonIsoEmPhi); 
   SetBranchAddress(tree, "nonIsoEmBx", className,  &nonIsoEmBx);
   SetBranchAddress(tree, "nCenJets", className,     &nCenJets);
   SetBranchAddress(tree, "cenJetEt", className,     &cenJetEt);
   SetBranchAddress(tree, "cenJetEta", className,    &cenJetEta);
   SetBranchAddress(tree, "cenJetPhi", className,    &cenJetPhi);  
   SetBranchAddress(tree, "cenJetBx", className,     &cenJetBx);
   SetBranchAddress(tree, "nFwdJets", className,     &nFwdJets);
   SetBranchAddress(tree, "fwdJetEt", className,     &fwdJetEt);
   SetBranchAddress(tree, "fwdJetEta", className,    &fwdJetEta);
   SetBranchAddress(tree, "fwdJetPhi", className,    &fwdJetPhi); 
   SetBranchAddress(tree, "fwdJetBx",  className,    &fwdJetBx);
   SetBranchAddress(tree, "nTauJets", className,     &nTauJets);
   SetBranchAddress(tree, "tauJetEt", className,     &tauJetEt);
   SetBranchAddress(tree, "tauJetEta", className,    &tauJetEta);
   SetBranchAddress(tree, "tauJetPhi", className,    &tauJetPhi); 
   SetBranchAddress(tree, "tauJetBx",  className,    &tauJetBx);
   SetBranchAddress(tree, "nMuons", className,       &nMuons);
   SetBranchAddress(tree, "muonEt", className,       &muonEt);
   SetBranchAddress(tree, "muonEta", className,      &muonEta);
   SetBranchAddress(tree, "muonPhi", className,      &muonPhi);
   SetBranchAddress(tree, "muonChg", className,      &muonChg);
   SetBranchAddress(tree, "muonIso", className,      &muonIso);
   SetBranchAddress(tree, "muonFwd", className,      &muonFwd);
   SetBranchAddress(tree, "muonMip", className,      &muonMip);
   SetBranchAddress(tree, "muonRPC", className,      &muonRPC); 
   SetBranchAddress(tree, "muonBx",  className,      &muonBx);
   SetBranchAddress(tree, "hfEtSum", className,      &hfEtSum);
   SetBranchAddress(tree, "hfBitCnt", className,     &hfBitCnt); 
   SetBranchAddress(tree, "hfBx", className,         &hfBx);
   SetBranchAddress(tree, "met", className,          &met);
   SetBranchAddress(tree, "metPhi", className,       &metPhi);
   SetBranchAddress(tree, "metsBx", className,       &metsBx);
   SetBranchAddress(tree, "mht", className,          &mht);
   SetBranchAddress(tree, "mhtPhi", className,       &mhtPhi);  
   SetBranchAddress(tree, "mhtsBx", className,       &mhtsBx);
   SetBranchAddress(tree, "et", className,           &et);
   SetBranchAddress(tree, "ht", className,           &ht);
}


void L1Analysis::L1AnalysisL1Extra::print()
{
}

bool L1Analysis::L1AnalysisL1Extra::check()
{
  bool test=true;
  return test;
}


#endif


