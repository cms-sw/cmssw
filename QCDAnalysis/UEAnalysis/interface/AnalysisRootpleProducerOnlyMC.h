#ifndef AnalysisRootpleProducerOnlyMC_H
#define AnalysisRootpleProducerOnlyMC_H

#include <iostream>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/Common/interface/Handle.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
//#include <PhysicsTools/UtilAlgos/interface/TFileService.h>
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class AnalysisRootpleProducerOnlyMC : public edm::EDAnalyzer
{
  
public:
  
  explicit AnalysisRootpleProducerOnlyMC( const edm::ParameterSet& ) ;
  virtual ~AnalysisRootpleProducerOnlyMC() {} 

  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  
  void fillEventInfo(int);
  void store();

private:
  
  edm::InputTag mcEvent; // label of MC event
  edm::InputTag genJetCollName; // label of Jet made with MC particles
  edm::InputTag chgJetCollName; // label of Jet made with only charged MC particles
  edm::InputTag chgGenPartCollName; // label of charged MC particles
  edm::InputTag gammaGenPartCollName; // label of gamma

  edm::Handle< edm::HepMCProduct              > EvtHandle        ;
  edm::Handle< std::vector<reco::GenParticle>  > CandHandleMC     ;
  edm::Handle< reco::GenJetCollection         > GenJetsHandle    ;
  edm::Handle< reco::GenJetCollection         > ChgGenJetsHandle ;
  edm::Handle< std::vector<reco::GenParticle> > GammaHandleMC    ;
 
  bool usegammaGen;
  
  float piG;

  edm::Service<TFileService> fs;

  TTree* AnalysisTree;

  int EventKind;
  
  TClonesArray* MonteCarlo;
  TClonesArray* InclusiveJet;
  TClonesArray* ChargedJet;
  TClonesArray* MCGamma;
  
};

#endif
