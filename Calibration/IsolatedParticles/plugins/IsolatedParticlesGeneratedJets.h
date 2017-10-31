#ifndef CalibrationIsolatedParticlesGeneratedJets_h
#define CalibrationIsolatedParticlesGeneratedJets_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "RecoJets/JetProducers/interface/JetMatchingTools.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

class IsolatedParticlesGeneratedJets : public edm::EDAnalyzer {

public:
  explicit IsolatedParticlesGeneratedJets(const edm::ParameterSet&);
  ~IsolatedParticlesGeneratedJets() override;

private:
  void beginJob() override ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;

  void    BookHistograms();
  void    clearTreeVectors();
  
  bool             debug;
  edm::Service<TFileService> fs;
  TTree            *tree;

  edm::EDGetTokenT<reco::GenJetCollection>      tok_jets_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_parts_;

  std::vector<int>    *t_gjetN;
  std::vector<double> *t_gjetE, *t_gjetPt, *t_gjetEta, *t_gjetPhi;
  std::vector< std::vector<double> > *t_jetTrkP;
  std::vector< std::vector<double> > *t_jetTrkPt;
  std::vector< std::vector<double> > *t_jetTrkEta;
  std::vector< std::vector<double> > *t_jetTrkPhi;
  std::vector< std::vector<double> > *t_jetTrkPdg;
  std::vector< std::vector<double> > *t_jetTrkCharge;

};

#endif
