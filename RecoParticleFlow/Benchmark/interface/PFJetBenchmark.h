#ifndef RecoParticleFlow_Benchmark_PFJetBenchmark_h
#define RecoParticleFlow_Benchmark_PFJetBenchmark_h

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


#include <string>

class PFJetBenchmark;

class TFile;
class TH1F;
class TH2F;

class PFJetBenchmark {

public:

  PFJetBenchmark();
  virtual ~PFJetBenchmark();

  void setup(
      std::string Filename,
	  bool debug, 
	  bool PlotAgainstReco, 
	  double deltaRMax=0.1  
	  );
  void process(const reco::PFJetCollection& , const reco::GenJetCollection& );
  void write();
  void gettrue (const reco::GenJet* truth , double& true_chargedEnergy, double& true_emEnergy);
  //void printPFJet (const reco::PFJet*);
  void printGenJet (const reco::GenJet*);
  // Data members
  double deltaEtMax_;
  double deltaChargedEnergyMax_;
  double deltaEmEnergyMax_; 

private:

  TFile *file_;

  TH1F *hDeltaEt;
  TH1F *hDeltaEch;
  TH1F *hDeltaEem;
  TH2F *hDeltaEtvsEt;
  TH2F *hDeltaEtOverEtvsEt;
  TH2F *hDeltaEtvsEta;
  TH2F *hDeltaEtOverEtvsEta;

  TH1F *hDeltaR;
  TH2F *hDeltaRvsEt;

protected:

  PFBenchmarkAlgo *algo_;
  bool debug_;
  bool PlotAgainstReco_;
  double deltaRMax_;
};

#endif // RecoParticleFlow_Benchmark_PFJetBenchmark_h
