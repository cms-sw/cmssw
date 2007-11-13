#ifndef RecoParticleFlow_PFProducer_TauBenchmarkAnalyzer
#define RecoParticleFlow_PFProducer_TauBenchmarkAnalyzer


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetfwd.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <string>

class PFBenchmarkAlgo;
class TH1F;
class TFile;

class TauBenchmarkAnalyzer : public edm::EDAnalyzer {
 public:
  explicit TauBenchmarkAnalyzer(const edm::ParameterSet&);
  ~TauBenchmarkAnalyzer();


 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  
  edm::Handle<reco::GenJetCollection> genJets_;
  edm::Handle<reco::CaloJetCollection> caloJets_;
  edm::Handle<reco::PFJetCollection> pfJets_;
  edm::Handle<edm::HepMCProduct> hepMC_; 

  PFBenchmarkAlgo *benchmark;
  std::string outputRootFileName_;
  std::string caloJetsLabel_;
  std::string pfJetsLabel_;
  std::string genJetsLabel_;
  TH1F *h_deltaETvisible_EHT_GEN_;
  TH1F *h_deltaETvisible_PF_GEN_;
  TFile *file_;
};

#endif
