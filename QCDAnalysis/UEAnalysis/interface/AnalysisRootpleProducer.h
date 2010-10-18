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
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"



// access trigger results
//#include <FWCore/Framework/interface/TriggerNames.h>
#include <FWCore/Common/interface/TriggerNames.h>
#include <DataFormats/Common/interface/TriggerResults.h>
#include <DataFormats/HLTReco/interface/TriggerEvent.h> 
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>

class TTree;
class TFile;
class TObject;



class AnalysisRootpleProducer : public edm::EDAnalyzer
{
  

public:
  
  //
  explicit AnalysisRootpleProducer( const edm::ParameterSet& ) ;
  virtual ~AnalysisRootpleProducer() {} // no need to delete ROOT stuff
  // as it'll be deleted upon closing TFile

  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  
  void fillEventInfo(int);
  void store();

private:
  
  bool onlyRECO;

  edm::InputTag mcEvent; // label of MC event
  edm::InputTag genJetCollName; // label of Jet made with MC particles
  edm::InputTag chgJetCollName; // label of Jet made with only charged MC particles
  edm::InputTag chgGenPartCollName; // label of charged MC particles
  edm::InputTag tracksJetCollName;
  edm::InputTag recoCaloJetCollName;
  edm::InputTag tracksCollName;
  edm::InputTag triggerResultsTag;
  edm::InputTag triggerEventTag;
  edm::InputTag genEventScaleTag;

  edm::Handle< double              > genEventScaleHandle;
  edm::Handle< edm::HepMCProduct        > EvtHandle ;
  edm::Handle< std::vector<reco::GenParticle> > CandHandleMC ;
  edm::Handle< reco::GenJetCollection    > GenJetsHandle ;
  edm::Handle< reco::GenJetCollection    > ChgGenJetsHandle ;
  //  edm::Handle< CandidateCollection > CandHandleRECO ;
  edm::Handle< edm::View<reco::Candidate> > CandHandleRECO ;
  edm::Handle< reco::TrackJetCollection  > TracksJetsHandle ;
  edm::Handle< reco::CaloJetCollection   > RecoCaloJetsHandle ;
  edm::Handle< edm::TriggerResults  > triggerResults;
  edm::Handle< trigger::TriggerEvent  > triggerEvent;

  //  edm::Handle<TriggerFilterObjectWithRefs> hltFilter; // not used at the moment: can access objects that fired the trigger
  edm::TriggerNames triggerNames;

  edm::Service<TFileService> fs;

  float piG;

  TTree* AnalysisTree;

  int EventKind;

  TClonesArray* Parton;
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
struct Vertex
{
  Int_t   npv;
  Double_t pvx[10];
  Double_t pvxErr[10];
  Double_t pvy[10];
  Double_t pvyErr[10];
  Double_t pvz[10];
  Double_t pvzErr[10];
  Double_t pvchi2[10];
  int   pvntk[10];
}vertex_;
 
struct TrackExtraUE
{
 Double_t  pvtkp[5000];
 Double_t pvtkpt[5000];
 Double_t pvtketa[5000];
 Double_t pvtkphi[5000];
 Double_t pvtknhit[5000];
 Double_t pvtkchi2norm[5000];
 Double_t pvtkd0[5000];
 Double_t pvtkd0Err[5000];
 Double_t pvtkdz[5000];
 Double_t pvtkdzErr[5000];
}trackextraue_;
 

struct TrackinJet
{
  Int_t tkn[100];
 Double_t tkp[5000];
 Double_t tkpt[5000];
 Double_t tketa[5000];
 Double_t tkphi[5000];
 Double_t tknhit[5000];
 Double_t tkchi2norm[5000];
 Double_t tkd0[5000];
 Double_t tkd0Err[5000];
 Double_t tkdz[5000];
 Double_t tkdzErr[5000];
}trackinjet_;
 

 std::vector<int>  pdgidList;

};

#endif
