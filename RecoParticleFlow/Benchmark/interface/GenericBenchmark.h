#ifndef RecoParticleFlow_Benchmark_GenericBenchmark_h
#define RecoParticleFlow_Benchmark_GenericBenchmark_h

//COLIN: necessary?
#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"


#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>

class DQMStore; // CMSSW_2_X_X

class BenchmarkTree;

class GenericBenchmark{

 public:

  GenericBenchmark();
  virtual ~GenericBenchmark();

  void setup(DQMStore *DQM = NULL, 
	     bool PlotAgainstReco_=true, 
	     float minDeltaEt = -100., float maxDeltaEt = 50., 
	     float minDeltaPhi = -0.5, float maxDeltaPhi = 0.5);
  //void setup(DQMStore *DQM = NULL, bool PlotAgainstReco_=true, float minDeltaEt = -200., float maxDeltaEt = 200., float minDeltaPhi = -3.2, float maxDeltaPhi = 3.2);
  //void setup(DQMStore *DQM = NULL, bool PlotAgainstReco_=true, float minDeltaEt, float maxDeltaEt, float minDeltaPhi, float maxDeltaPhi);

  void fill(const edm::View<reco::Candidate> *RecoCollection, 
	    const edm::View<reco::Candidate> *GenCollection,
	    bool startFromGen=false, 
	    bool PlotAgainstReco =true, 
	    bool onlyTwoJets = false, 
	    double recPt_cut = -1., 
	    double minEta_cut = -1., 
	    double maxEta_cut = -1., 
	    double deltaR_cut = -1.);

  void write(std::string Filename);

  void fillHistos( const reco::Candidate* genParticle,
		   const reco::Candidate* recParticle,
		   double deltaR_cut,
		   bool plotAgainstReco); 

  void setfile(TFile *file);

 private:
  
  bool accepted(const reco::Candidate* particle,
		double ptCut,
		double minEtaCut,
		double maxEtaCut ) const;
    
  TFile *file_;

  TH1F *hDeltaEt;
  TH1F *hDeltaEx;
  TH1F *hDeltaEy;
  TH2F *hDeltaEtvsEt;
  TH2F *hDeltaEtOverEtvsEt;
  TH2F *hDeltaEtvsEta;
  TH2F *hDeltaEtOverEtvsEta;
  TH2F *hDeltaEtvsPhi;
  TH2F *hDeltaEtOverEtvsPhi;
  TH2F *hDeltaEtvsDeltaR;
  TH2F *hDeltaEtOverEtvsDeltaR;

  TH2F *hEtRecvsEt;

  TH1F *hDeltaEta;
  TH2F *hDeltaEtavsEt;
  TH2F *hDeltaEtavsEta;

  TH1F *hDeltaPhi;
  TH2F *hDeltaPhivsEt;
  TH2F *hDeltaPhivsEta;

  TH1F *hDeltaR;
  TH2F *hDeltaRvsEt;
  TH2F *hDeltaRvsEta;

  TH1F *hNRec;


  TH1F *hEtGen;
  TH1F *hEtaGen;
  TH1F *hPhiGen;

  TH1F *hNGen;

  TH1F *hEtSeen;
  TH1F *hEtaSeen;
  TH1F *hPhiSeen;

  TH1F *hEtRec;
  TH1F *hExRec;
  TH1F *hEyRec;
  TH1F *hPhiRec;

  BenchmarkTree*  tree_;

  bool fillFunctionHasBeenUsed_;

 protected:

  DQMStore *dbe_;
  PFBenchmarkAlgo *algo_;

};


#endif // RecoParticleFlow_Benchmark_GenericBenchmark_h
