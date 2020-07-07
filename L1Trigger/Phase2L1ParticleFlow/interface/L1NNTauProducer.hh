#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOW_L1NNTAU_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOW_L1NNTAU_H

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/TauNNId.h"

using namespace l1t;

class L1NNTauProducer : public edm::stream::EDProducer<> {
public:
  explicit L1NNTauProducer(const edm::ParameterSet &);
  ~L1NNTauProducer() override;

private:
  std::unique_ptr<TauNNId> fTauNNId_;
  void addTau(const l1t::PFCandidate &iCand,
              const l1t::PFCandidateCollection &iParts,
              std::unique_ptr<PFTauCollection> &outputTaus);
  float deltaR(const l1t::PFCandidate &iPart1, const l1t::PFCandidate &iPart2);
  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

  double fSeedPt_;
  double fConeSize_;
  double fTauSize_;
  int fMaxTaus_;
  int fNParticles_;
  edm::EDGetTokenT<vector<l1t::PFCandidate> > fL1PFToken_;
};

#endif
