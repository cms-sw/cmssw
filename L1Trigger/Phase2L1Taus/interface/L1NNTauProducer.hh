#ifndef L1TRIGGER_PHASE2L1TAU_L1NNTAU_H
#define L1TRIGGER_PHASE2L1TAU_L1NNTAU_H

#include <iostream>
#include <vector>
#include <TLorentzVector.h>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Phase2L1ParticleFlow/interface/PFTau.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1Taus/interface/TauNNId.h"

using namespace l1t;

class L1NNTauProducer : public edm::stream::EDProducer<> {
public:
  explicit L1NNTauProducer(const edm::ParameterSet&);
  ~L1NNTauProducer();

private:
  TauNNId *fTauNNId;
  void addTau(l1t::PFCandidate &iCand,const l1t::PFCandidateCollection &iParts, std::unique_ptr<PFTauCollection> &outputTaus);
  float deltaR(auto iPart1, auto iPart2);
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup ) override;

  double fSeedPt;
  double fConeSize;
  double fTauSize;
  int fMaxTaus;
  int fNParticles;
  edm::EDGetTokenT< vector<l1t::PFCandidate> > fL1PFToken;
};


#endif
