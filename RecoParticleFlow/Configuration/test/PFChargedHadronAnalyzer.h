#ifndef RecoParticleFlow_PFPatProducer_PFChargedHadronAnalyzer_
#define RecoParticleFlow_PFPatProducer_PFChargedHadronAnalyzer_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"

#include <TFile.h>
#include <TTree.h>

/**\class PFChargedHadronAnalyzer 
\brief selects isolated charged hadrons from PF Charged Hadrons

\author Patrick Janot
\date   September 13, 2010
*/




class PFChargedHadronAnalyzer : public edm::EDAnalyzer {
 public:

  typedef reco::PFCandidateCollection::const_iterator CI;

  explicit PFChargedHadronAnalyzer(const edm::ParameterSet&);

  ~PFChargedHadronAnalyzer();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);

 private:
  

  /// PFCandidates in which we'll look for pile up particles 
  edm::InputTag   inputTagPFCandidates_;
  edm::InputTag   inputTagPFSimParticles_;
  
  /// Min pt for charged hadrons
  double ptMin_;
  
  /// Min p for charged hadrons
  double pMin_;

  /// Min hcal raw energy for charged hadrons
  double hcalMin_;
  
  /// Max ecal raw energy to define a MIP
  double ecalMax_;
  
  /// Min number of pixel hits for charged hadrons
  int nPixMin_;
  
  /// Min number of track hits for charged hadrons
  std::vector<int> nHitMin_;
  std::vector<double> nEtaMin_;
  
  // Number of tracks after cuts
  std::vector<unsigned int> nCh;
  std::vector<unsigned int> nEv;
  
  std::string outputfile_;
  TFile *tf1;
  TTree* s;
  
  float true_,p_,ecal_,hcal_,eta_,phi_;

  /// verbose ?
  bool   verbose_;



};

#endif
