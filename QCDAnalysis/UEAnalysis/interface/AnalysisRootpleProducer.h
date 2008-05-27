#ifndef AnalysisRootpleProducer_H
#define AnalysisRootpleProducer_H

#include <iostream>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/Common/interface/Handle.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <PhysicsTools/UtilAlgos/interface/TFileService.h>

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <TObjString.h>
#include <TClonesArray.h>

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// access trigger results
#include <FWCore/Framework/interface/TriggerNames.h>
#include <DataFormats/Common/interface/TriggerResults.h>
#include <DataFormats/HLTReco/interface/TriggerEvent.h> 
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>

using namespace edm;
using namespace reco;
using namespace trigger;
using std::vector;

class AnalysisRootpleProducer : public edm::EDAnalyzer
{
  
public:
  
  //
  explicit AnalysisRootpleProducer( const edm::ParameterSet& ) ;
  virtual ~AnalysisRootpleProducer() {} // no need to delete ROOT stuff
  // as it'll be deleted upon closing TFile
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;
  
  void fillEventInfo(int);
  void store();

private:
  
  bool onlyRECO;

  InputTag mcEvent; // label of MC event
  InputTag genJetCollName; // label of Jet made with MC particles
  InputTag chgJetCollName; // label of Jet made with only charged MC particles
  InputTag chgGenPartCollName; // label of charged MC particles
  InputTag tracksJetCollName;
  InputTag recoCaloJetCollName;
  InputTag tracksCollName;
  InputTag triggerResultsTag;
  InputTag triggerEventTag;

  Handle< double              > genEventScaleHandle;
  Handle< HepMCProduct        > EvtHandle ;
  Handle< vector<GenParticle> > CandHandleMC ;
  Handle< GenJetCollection    > GenJetsHandle ;
  Handle< GenJetCollection    > ChgGenJetsHandle ;
  Handle< CandidateCollection > CandHandleRECO ;
  Handle< BasicJetCollection  > TracksJetsHandle ;
  Handle< CaloJetCollection   > RecoCaloJetsHandle ;
  Handle< TriggerResults      > triggerResults;
  Handle< TriggerEvent        > triggerEvent;
  //  Handle<TriggerFilterObjectWithRefs> hltFilter; // not used at the moment: can access objects that fired the trigger
  TriggerNames triggerNames;

  edm::Service<TFileService> fs;

  float piG;

  TTree* AnalysisTree;

  int EventKind;
  
  TClonesArray* MonteCarlo;
  TClonesArray* InclusiveJet;
  TClonesArray* ChargedJet;
  TClonesArray* Track;
  TClonesArray* TracksJet;
  TClonesArray* CalorimeterJet;
  TClonesArray* acceptedTriggers;

  double genEventScale;
};

#endif
