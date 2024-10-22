#ifndef CALO_MCTRUTH_TREE_PRODUCER_H
#define CALO_MCTRUTH_TREE_PRODUCER_H

#include "TTree.h"
#include "TFile.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

//namespace cms
//{
class CaloMCTruthTreeProducer : public edm::one::EDAnalyzer<> {
public:
  explicit CaloMCTruthTreeProducer(edm::ParameterSet const& cfg);
  void beginJob() override;
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;
  void endJob() override;
  ~CaloMCTruthTreeProducer() override;

private:
  std::string histogramFile_;
  edm::EDGetTokenT<reco::CaloJetCollection> jets_;
  edm::EDGetTokenT<reco::GenJetCollection> genjets_;
  edm::EDGetTokenT<GenEventInfoProduct> gen_;
  TFile* file_;
  TTree* mcTruthTree_;
  float ptJet_, emfJet_, ptGen_, ptHat_, dR_, etaJet_, etaGen_, phiJet_, phiGen_;
  int rank_;
};
//}

#endif
