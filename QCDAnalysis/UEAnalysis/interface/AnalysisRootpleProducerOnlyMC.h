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
#include <CommonTools/UtilAlgos/interface/TFileService.h>

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
  void fillMCParticles(float, float, float, float);
  void fillInclusiveJet(float, float, float, float);
  void fillChargedJet(float, float, float, float);
  void store();

private:

  edm::EDGetTokenT< edm::HepMCProduct         > mcEventToken; // label of MC event
  edm::EDGetTokenT< reco::GenJetCollection    > genJetCollToken; // label of Jet made with MC particles
  edm::EDGetTokenT< reco::GenJetCollection    > chgJetCollToken; // label of Jet made with only charged MC particles
  edm::EDGetTokenT< std::vector<reco::GenParticle> > chgGenPartCollToken; // label of charged MC particles

  edm::Handle< edm::HepMCProduct         > EvtHandle        ;
  edm::Handle< std::vector<reco::GenParticle> > CandHandleMC     ;
  edm::Handle< reco::GenJetCollection    > GenJetsHandle    ;
  edm::Handle< reco::GenJetCollection    > ChgGenJetsHandle ;


  float piG;

  edm::Service<TFileService> fs;

  TTree* AnalysisTree;

  static const int NMCPMAX = 10000;
  static const int NTKMAX = 10000;
  static const int NIJMAX = 10000;
  static const int NCJMAX = 10000;
  static const int NTJMAX = 10000;
  static const int NEHJMAX = 10000;

  int EventKind,NumberMCParticles,NumberTracks,NumberInclusiveJet,NumberChargedJet,NumberTracksJet,NumberCaloJet;

  float MomentumMC[NMCPMAX],TransverseMomentumMC[NMCPMAX],EtaMC[NMCPMAX],PhiMC[NMCPMAX];
  float MomentumTK[NTKMAX],TransverseMomentumTK[NTKMAX],EtaTK[NTKMAX],PhiTK[NTKMAX];
  float MomentumIJ[NIJMAX],TransverseMomentumIJ[NIJMAX],EtaIJ[NIJMAX],PhiIJ[NIJMAX];
  float MomentumCJ[NCJMAX],TransverseMomentumCJ[NCJMAX],EtaCJ[NCJMAX],PhiCJ[NCJMAX];
  float MomentumTJ[NTJMAX],TransverseMomentumTJ[NTJMAX],EtaTJ[NTJMAX],PhiTJ[NTJMAX];
  float MomentumEHJ[NEHJMAX],TransverseMomentumEHJ[NEHJMAX],EtaEHJ[NEHJMAX],PhiEHJ[NEHJMAX];

  TClonesArray* MonteCarlo;
  TClonesArray* InclusiveJet;
  TClonesArray* ChargedJet;

};

#endif
