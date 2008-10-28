#ifndef RecoParticleFlow_Benchmark_GenericBenchmark_h
#define RecoParticleFlow_Benchmark_GenericBenchmark_h

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>

class DQMStore; // CMSSW_2_X_X
//class DaqMonitorBEInterface; // CMSSW_1_X_X

class GenericBenchmark{
public:

  GenericBenchmark();
  virtual ~GenericBenchmark();

  void setup(DQMStore *DQM = NULL, bool PlotAgainstReco_=true); // CMSSW_2_X_X
  //void setup(DaqMonitorBEInterface *DQM = NULL); // CMSSW_1_X_X
  void fill(const edm::View<reco::Candidate> *RecoCollection, const edm::View<reco::Candidate> *GenCollection, bool PlotAgainstReco =true, bool onlyTwoJets = false, double recPt_cut = -1., double maxEta_cut = -1., double deltaR_cut = -1.);
  void write(std::string Filename);

private:

  TFile *file_;

  TH1F *hDeltaEt;
  TH2F *hDeltaEtvsEt;
  TH2F *hDeltaEtOverEtvsEt;
  TH2F *hDeltaEtvsEta;
  TH2F *hDeltaEtOverEtvsEta;
  TH2F *hDeltaEtvsPhi;
  TH2F *hDeltaEtOverEtvsPhi;
  TH2F *hDeltaEtvsDeltaR;
  TH2F *hDeltaEtOverEtvsDeltaR;

  TH1F *hDeltaEta;
  TH2F *hDeltaEtavsEt;
  TH2F *hDeltaEtaOverEtavsEt; // ms: propose remove
  TH2F *hDeltaEtavsEta;
  TH2F *hDeltaEtaOverEtavsEta; // ms: propose remove
  TH2F *hDeltaEtavsPhi; // ms: propose remove
  TH2F *hDeltaEtaOverEtavsPhi; // ms: propose remove

  TH1F *hDeltaPhi;
  TH2F *hDeltaPhivsEt;
  TH2F *hDeltaPhiOverPhivsEt; // ms: propose remove
  TH2F *hDeltaPhivsEta;
  TH2F *hDeltaPhiOverPhivsEta; // ms: propose remove
  TH2F *hDeltaPhivsPhi; // ms: propose remove
  TH2F *hDeltaPhiOverPhivsPhi; // ms: propose remove

  TH1F *hDeltaR;
  TH2F *hDeltaRvsEt;
  TH2F *hDeltaRvsEta;
  TH2F *hDeltaRvsPhi; // ms: propose remove

protected:

  DQMStore *dbe_;
  PFBenchmarkAlgo *algo_;

};


#endif // RecoParticleFlow_Benchmark_GenericBenchmark_h
