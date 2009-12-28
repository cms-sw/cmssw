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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <PhysicsTools/UtilAlgos/interface/TFileService.h>

#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include <TVector.h>
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"



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
  InputTag genEventScaleTag;

  Handle< double              > genEventScaleHandle;
  Handle< HepMCProduct        > EvtHandle ;
  Handle< vector<GenParticle> > CandHandleMC ;
  Handle< GenJetCollection    > GenJetsHandle ;
  Handle< GenJetCollection    > ChgGenJetsHandle ;
  //  Handle< CandidateCollection > CandHandleRECO ;
  Handle< edm::View<reco::Candidate> > CandHandleRECO ;
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
  TClonesArray* MonteCarlo2;
  TClonesArray* InclusiveJet;
  TClonesArray* ChargedJet;
  TClonesArray* Track;
  TClonesArray* AssVertex;
  TClonesArray* TracksJet;
  TClonesArray* CalorimeterJet;
  TClonesArray* acceptedTriggers;

  double genEventScale;
  //info sull'evento 
  int eventNum;
  int lumiBlock;
  int runNumber;
  int bx;

  //tracks with vertex
  Int_t   m_npv;
  Double_t m_pvx[10];
  Double_t m_pvxErr[10];
  Double_t m_pvy[10];
  Double_t m_pvyErr[10];
  Double_t m_pvz[10];
  Double_t m_pvzErr[10];
  Double_t m_pvchi2[10];
  int   m_pvntk[10];
  
  //Double_t  m_pvtkp[5000];
  //Double_t m_pvtkpt[5000];
  //Double_t m_pvtketa[5000];
  //Double_t m_pvtkphi[5000];
  //Double_t m_pvtknhit[5000];
  //Double_t m_pvtkchi2norm[5000];
  //Double_t m_pvtkd0[5000];
  //Double_t m_pvtkd0Err[5000];
  //Double_t m_pvtkdz[5000];
  //Double_t m_pvtkdzErr[5000];

  //int  m_ntk;
  //Double_t  m_tkp[5000];
  //Double_t m_tkpt[5000];
  //Double_t m_tketa[5000];
  //Double_t m_tkphi[5000];
  //Double_t m_tknhit[5000];
  //Double_t m_tkchi2norm[5000];
  //Double_t m_tkd0[5000];
  //Double_t m_tkd0Err[5000];
  //Double_t m_tkdz[5000];
  //Double_t m_tkdzErr[5000];
  vector<int>  pdgidList;

};

#endif
