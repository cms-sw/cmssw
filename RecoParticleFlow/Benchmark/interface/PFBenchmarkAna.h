#ifndef RecoParticleFlow_Benchmark_PFBenchmarkAna_h
#define RecoParticleFlow_Benchmark_PFBenchmarkAna_h

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include <string>

class DaqMonitorBEInterface;
class PFBenchmarkAlgo;

class TFile;
class TH1F;
class TH2F;

class PFBenchmarkAna {
public:

  PFBenchmarkAna();
  virtual ~PFBenchmarkAna();

  void setup(DaqMonitorBEInterface *DQM = NULL);
  void fill(const reco::PFCandidateCollection *, const reco::CandidateCollection *, bool PlotAgainstReco = true);
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
  TH2F *hDeltaEtaOverEtavsEt;
  TH2F *hDeltaEtavsEta;
  TH2F *hDeltaEtaOverEtavsEta;
  TH2F *hDeltaEtavsPhi;
  TH2F *hDeltaEtaOverEtavsPhi;

  TH1F *hDeltaPhi;
  TH2F *hDeltaPhivsEt;
  TH2F *hDeltaPhiOverPhivsEt;
  TH2F *hDeltaPhivsEta;
  TH2F *hDeltaPhiOverPhivsEta;
  TH2F *hDeltaPhivsPhi;
  TH2F *hDeltaPhiOverPhivsPhi;

  TH1F *hDeltaR;
  TH2F *hDeltaRvsEt;
  TH2F *hDeltaRvsEta;
  TH2F *hDeltaRvsPhi;

protected:

  DaqMonitorBEInterface *dbe_;
  PFBenchmarkAlgo *algo_;

};

#endif // RecoParticleFlow_Benchmark_PFBenchmarkAna_h
