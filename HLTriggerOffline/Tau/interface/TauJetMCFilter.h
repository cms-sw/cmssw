#ifndef TauJetMCFilter_H
#define TauJetMCFilter_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"

#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms
#include "TH1.h" // RooT histogram class

#include <string>
#include <vector>
#include <set>
class TauJetMCFilter: public edm::EDFilter {
 public:
  explicit TauJetMCFilter(const edm::ParameterSet&);
  ~TauJetMCFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;


 private:
  typedef std::vector<std::string> vstring;
  edm::InputTag genParticles;
  double mEtaMin, mEtaMax, mEtTau,mEtaElecMax,mPtElec,mEtaMuonMax,mPtMuon;
  vstring  mincludeList;
  //int mn_taujet,mn_elec,mn_muon;
  typedef std::vector< HepMC::GenParticle * > GenPartVect;
  typedef std::vector< HepMC::GenParticle * >::const_iterator GenPartVectIt;
  HepMC::GenParticle * findParticle(const GenPartVect genPartVect, const int requested_id) ;


  // Printout efficiency statistics & histos
  bool _fillHistos;
  bool _doPrintOut;

  TH1F* h_ElecEt; 
  TH1F* h_ElecEta; 
  TH1F* h_ElecPhi; 

  TH1F* h_MuonPt; 
  TH1F* h_MuonEta; 
  TH1F* h_MuonPhi; 

  TH1F* h_TauEt; 
  TH1F* h_TauEta; 
  TH1F* h_TauPhi; 

  int _nEvents;
  int _nPassedElecEtaCut;
  int _nPassedElecEtCut;
  int _nPassedMuonEtaCut;
  int _nPassedMuonPtCut;
  int _nPassedTauEtaCut;
  int _nPassedTauEtCut;

  int _nPassednElecCut;
  int _nPassednMuonCut;
  int _nPassednTauCut;

  int _nPassedAllCuts;


};
#endif
