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
  ~IsolatedParticlesGeneratedJets();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void    BookHistograms();
  void    clearTreeVectors();
  
  bool             debug;
  edm::InputTag    jetSrc, partSrc;
  edm::Service<TFileService> fs;
  TTree            *tree;

  std::vector<int> *t_gjetN;

  std::vector<double> *t_gjetE, *t_gjetPt, *t_gjetEta, *t_gjetPhi;

  std::vector< std::vector<double> > *t_jetTrkP;
  std::vector< std::vector<double> > *t_jetTrkPt;
  std::vector< std::vector<double> > *t_jetTrkEta;
  std::vector< std::vector<double> > *t_jetTrkPhi;
  std::vector< std::vector<double> > *t_jetTrkPdg;
  std::vector< std::vector<double> > *t_jetTrkCharge;

};

#endif
