#ifndef RecoParticleFlow_Benchmark_PFBenchmarkAlgo
#define RecoParticleFlow_Benchmark_PFBenchmarkAlgo


#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetfwd.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <string>

class TH1F;
class TFile;

class PFBenchmarkAlgo
{
 public:
  PFBenchmarkAlgo(std::string);
  ~PFBenchmarkAlgo();
  void doBenchmark();
  void createPlots();
  void setOutputRootFileName(std::string);
  void setGenJets(edm::Handle<reco::GenJetCollection>);
  void setCaloJets(edm::Handle<reco::CaloJetCollection>);
  void setPfJets(edm::Handle<reco::PFJetCollection>);
  void setHepMC(edm::Handle<edm::HepMCProduct>);

 private:
  std::string outputRootFileName_;
  edm::Handle<reco::GenJetCollection> genJets_;
  edm::Handle<reco::CaloJetCollection> caloJets_;
  edm::Handle<reco::PFJetCollection> pfJets_;
  edm::Handle<edm::HepMCProduct> hepMC_; 

  TFile *file_;
  
  TH1F *h_deltaETvisible_EHT_GEN_;
  TH1F *h_deltaETvisible_PF_GEN_;

  TH1F* h_deltaETvisible_MCHEPMC_PF_;	                      
  TH1F* h_deltaETvisible_MCHEPMC_EHT_;
  
  TH1F* h_deltaEtvsEt_MCHEPMC_PF_;
  TH1F* h_deltaEtvsEt_MCHEPMC_EHT_;
  TH1F* n_deltaEtvsEt;

  TH1F* h_deltaEtvsEta_MCHEPMC_PF_;
  TH1F* h_deltaEtvsEta_MCHEPMC_EHT_;
  TH1F* n_deltaEtvsEta;

  TH1F* h_deltaEtDivEtvsEt_MCHEPMC_PF_;
  TH1F* h_deltaEtDivEtvsEt_MCHEPMC_EHT_;
  TH1F* n_deltaEtDivEtvsEt;

  TH1F* h_deltaEtDivEtrecvsEt_MCHEPMC_PF_;
  TH1F* h_deltaEtDivEtrecvsEt_MCHEPMC_EHT_;//
  TH1F* n_deltaEtDivEtrecvsEt;
 
  TH1F* h_deltaEtDivEtvsEta_MCHEPMC_PF_;
  TH1F* h_deltaEtDivEtvsEta_MCHEPMC_EHT_;
  TH1F* n_deltaEtDivEtvsEta;
 
  TH1F* h_deltaEtDivEtrecvsEta_MCHEPMC_PF_;
  TH1F* h_deltaEtDivEtrecvsEta_MCHEPMC_EHT_;//
  TH1F* n_deltaEtDivEtrecvsEta;
 
  TH1F* h_deltaEta_MCHEPMC_PF_;
  TH1F* h_deltaEta_MCHEPMC_EHT_;
  
  TH1F* h_deltaEtavsPt_MCHEPMC_PF_;
  TH1F* h_deltaEtavsPt_MCHEPMC_EHT_;
  TH1F* n_deltaEtavsPt;
 
  TH1F* h_deltaEtavsEta_MCHEPMC_PF_;
  TH1F* h_deltaEtavsEta_MCHEPMC_EHT_;
  TH1F* n_deltaEtavsEta;
 
  TH1F* h_deltaPhi_MCHEPMC_PF_;
  TH1F* h_deltaPhi_MCHEPMC_EHT_;
  TH1F* n_deltaPhi;

  TH1F* h_deltaPhivsPt_MCHEPMC_PF_;
  TH1F* h_deltaPhivsPt_MCHEPMC_EHT_;
  TH1F* n_deltaPhivsPt;
 
  TH1F* h_deltaPhivsEta_MCHEPMC_PF_;
  TH1F* h_deltaPhivsEta_MCHEPMC_EHT_;
  TH1F* n_deltaPhivsEta;
  
  // ABC____________________
  
  TH1F* h_deltaETDivTrue_MCHEPMC_PF_;
  TH1F* h_deltaETDivTrue_MCHEPMC_EHT_;
  
  TH1F* h_deltaETDivReco_MCHEPMC_PF_;
  TH1F* h_deltaETDivReco_MCHEPMC_EHT_;
  
  TH1F* h_deltaEtaDivReco_MCHEPMC_PF_;
  TH1F* h_deltaEtaDivReco_MCHEPMC_EHT_;
  
  TH1F* h_deltaPhiDivReco_MCHEPMC_PF_;
  TH1F* h_deltaPhiDivReco_MCHEPMC_EHT_;
  
  TH1F* h_ErecDivEtrue_MCHEPMC_PF_;
  TH1F* h_ErecDivEtrue_MCHEPMC_EHT_;
  TH1F* n_ErecDivEtrue;

};

#endif
