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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TH1F.h"
#include "TH2F.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include <TFile.h>
#include <vector>

class PFJetBenchmark {
public:
  PFJetBenchmark();
  virtual ~PFJetBenchmark();

  void setup(std::string Filename,
             bool debug,
             bool plotAgainstReco = false,
             bool onlyTwoJets = true,
             double deltaRMax = 0.1,
             std::string benchmarkLabel_ = "ParticleFlow",
             double recPt = -1,
             double maxEta = -1,
             DQMStore *dbe_store = nullptr);
  void process(const reco::PFJetCollection &, const reco::GenJetCollection &);
  void gettrue(const reco::GenJet *truth,
               double &true_ChargedHadEnergy,
               double &true_NeutralHadEnergy,
               double &true_NeutralEmEnergy);
  void printPFJet(const reco::PFJet *);
  void printGenJet(const reco::GenJet *);
  double resPtMax() const { return resPtMax_; };
  double resChargedHadEnergyMax() const { return resChargedHadEnergyMax_; };
  double resNeutralHadEnergyMax() const { return resNeutralHadEnergyMax_; };
  double resNeutralEmEnergyMax() const { return resNeutralEmEnergyMax_; };
  //  void save();
  void write();

private:
  TFile *file_;

  // histograms
  // Jets inclusive  distributions  (Pt > 20 GeV)
  TH1F *hNjets;
  TH1F *hjetsPt;
  TH1F *hjetsEta;
  TH2F *hRPtvsEta;
  TH2F *hDEtavsEta;
  TH2F *hDPhivsEta;
  TH2F *hRNeutvsEta;
  TH2F *hRNEUTvsEta;
  TH2F *hRNONLvsEta;
  TH2F *hRHCALvsEta;
  TH2F *hRHONLvsEta;
  TH2F *hRCHEvsEta;
  TH2F *hNCHvsEta;
  TH2F *hNCH0vsEta;
  TH2F *hNCH1vsEta;
  TH2F *hNCH2vsEta;
  TH2F *hNCH3vsEta;
  TH2F *hNCH4vsEta;
  TH2F *hNCH5vsEta;
  TH2F *hNCH6vsEta;
  TH2F *hNCH7vsEta;

  // delta Pt or E quantities for Barrel
  TH1F *hBRPt;
  TH1F *hBRPt20_40;
  TH1F *hBRPt40_60;
  TH1F *hBRPt60_80;
  TH1F *hBRPt80_100;
  TH1F *hBRPt100_150;
  TH1F *hBRPt150_200;
  TH1F *hBRPt200_250;
  TH1F *hBRPt250_300;
  TH1F *hBRPt300_400;
  TH1F *hBRPt400_500;
  TH1F *hBRPt500_750;
  TH1F *hBRPt750_1250;
  TH1F *hBRPt1250_2000;
  TH1F *hBRPt2000_5000;
  TH1F *hBRCHE;
  TH1F *hBRNHE;
  TH1F *hBRNEE;
  TH1F *hBRneut;
  TH2F *hBRPtvsPt;
  TH2F *hBRCHEvsPt;
  TH2F *hBRNHEvsPt;
  TH2F *hBRNEEvsPt;
  TH2F *hBRneutvsPt;
  TH2F *hBRNEUTvsP;
  TH2F *hBRNONLvsP;
  TH2F *hBRHCALvsP;
  TH2F *hBRHONLvsP;
  TH2F *hBDEtavsPt;
  TH2F *hBDPhivsPt;
  TH2F *hBNCHvsPt;
  TH1F *hBNCH;
  TH2F *hBNCH0vsPt;
  TH2F *hBNCH1vsPt;
  TH2F *hBNCH2vsPt;
  TH2F *hBNCH3vsPt;
  TH2F *hBNCH4vsPt;
  TH2F *hBNCH5vsPt;
  TH2F *hBNCH6vsPt;
  TH2F *hBNCH7vsPt;

  // delta Pt or E quantities for Endcap
  TH1F *hERPt;
  TH1F *hERPt20_40;
  TH1F *hERPt40_60;
  TH1F *hERPt60_80;
  TH1F *hERPt80_100;
  TH1F *hERPt100_150;
  TH1F *hERPt150_200;
  TH1F *hERPt200_250;
  TH1F *hERPt250_300;
  TH1F *hERPt300_400;
  TH1F *hERPt400_500;
  TH1F *hERPt500_750;
  TH1F *hERPt750_1250;
  TH1F *hERPt1250_2000;
  TH1F *hERPt2000_5000;
  TH1F *hERCHE;
  TH1F *hERNHE;
  TH1F *hERNEE;
  TH1F *hERneut;
  TH2F *hERPtvsPt;
  TH2F *hERCHEvsPt;
  TH2F *hERNHEvsPt;
  TH2F *hERNEEvsPt;
  TH2F *hERneutvsPt;
  TH2F *hERNEUTvsP;
  TH2F *hERNONLvsP;
  TH2F *hERHCALvsP;
  TH2F *hERHONLvsP;
  TH2F *hEDEtavsPt;
  TH2F *hEDPhivsPt;
  TH2F *hENCHvsPt;
  TH1F *hENCH;
  TH2F *hENCH0vsPt;
  TH2F *hENCH1vsPt;
  TH2F *hENCH2vsPt;
  TH2F *hENCH3vsPt;
  TH2F *hENCH4vsPt;
  TH2F *hENCH5vsPt;
  TH2F *hENCH6vsPt;
  TH2F *hENCH7vsPt;

  // delta Pt or E quantities for Forward
  TH1F *hFRPt;
  TH1F *hFRPt20_40;
  TH1F *hFRPt40_60;
  TH1F *hFRPt60_80;
  TH1F *hFRPt80_100;
  TH1F *hFRPt100_150;
  TH1F *hFRPt150_200;
  TH1F *hFRPt200_250;
  TH1F *hFRPt250_300;
  TH1F *hFRPt300_400;
  TH1F *hFRPt400_500;
  TH1F *hFRPt500_750;
  TH1F *hFRPt750_1250;
  TH1F *hFRPt1250_2000;
  TH1F *hFRPt2000_5000;
  TH1F *hFRCHE;
  TH1F *hFRNHE;
  TH1F *hFRNEE;
  TH1F *hFRneut;
  TH2F *hFRPtvsPt;
  TH2F *hFRCHEvsPt;
  TH2F *hFRNHEvsPt;
  TH2F *hFRNEEvsPt;
  TH2F *hFRneutvsPt;
  TH2F *hFRNEUTvsP;
  TH2F *hFRNONLvsP;
  TH2F *hFRHCALvsP;
  TH2F *hFRHONLvsP;
  TH2F *hFDEtavsPt;
  TH2F *hFDPhivsPt;
  TH2F *hFNCHvsPt;
  TH1F *hFNCH;
  TH2F *hFNCH0vsPt;
  TH2F *hFNCH1vsPt;
  TH2F *hFNCH2vsPt;
  TH2F *hFNCH3vsPt;
  TH2F *hFNCH4vsPt;
  TH2F *hFNCH5vsPt;
  TH2F *hFNCH6vsPt;
  TH2F *hFNCH7vsPt;

  std::string outputFile_;

protected:
  PFBenchmarkAlgo *algo_;
  bool debug_;
  bool plotAgainstReco_;
  bool onlyTwoJets_;
  double deltaRMax_;
  double resPtMax_;
  double resChargedHadEnergyMax_;
  double resNeutralHadEnergyMax_;
  double resNeutralEmEnergyMax_;
  double recPt_cut;
  double maxEta_cut;
  unsigned int entry_;
  DQMStore *dbe_;
};

#endif  // RecoParticleFlow_Benchmark_PFJetBenchmark_h
