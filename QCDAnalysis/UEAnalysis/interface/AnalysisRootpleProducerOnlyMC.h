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
#include <PhysicsTools/UtilAlgos/interface/TFileService.h>

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

using namespace edm;
using namespace reco;
using std::vector;

class AnalysisRootpleProducerOnlyMC : public edm::EDAnalyzer
{
  
public:
  
  explicit AnalysisRootpleProducerOnlyMC( const edm::ParameterSet& ) ;
  virtual ~AnalysisRootpleProducerOnlyMC() {} 

  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;
  
  void fillEventInfo(int);
  void store();

private:
  
  InputTag mcEvent; // label of MC event
  InputTag genJetCollName; // label of Jet made with MC particles
  InputTag chgJetCollName; // label of Jet made with only charged MC particles
  InputTag chgGenPartCollName; // label of charged MC particles
  InputTag gammaGenPartCollName; // label of gamma

  Handle< HepMCProduct        > EvtHandle        ;
  Handle< vector<GenParticle> > CandHandleMC     ;
  Handle< GenJetCollection    > GenJetsHandle    ;
  Handle< GenJetCollection    > ChgGenJetsHandle ;
  Handle< vector<GenParticle> > GammaHandleMC    ;

  
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
