#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <iostream>
using namespace edm;
using namespace std;
using namespace reco;

class DebugZMCTruth : public edm::EDAnalyzer {
public:
  DebugZMCTruth(const edm::ParameterSet& pset);
private:
  virtual void analyze(const Event& event, const EventSetup& setup) override;
  InputTag src_;
  EDGetTokenT<CandidateView> srcToken_;
  EDGetTokenT<GenParticleCollection> genParticlesToken_;
  EDGetTokenT<GenParticleMatch> matchToken_;
};

DebugZMCTruth::DebugZMCTruth(const ParameterSet& cfg) :
  src_(cfg.getParameter<InputTag>("src")),
  srcToken_(consumes<CandidateView>(src_)),
  genParticlesToken_(consumes<GenParticleCollection>(cfg.getParameter<InputTag>("genParticles"))),
  matchToken_(consumes<GenParticleMatch>(cfg.getParameter<InputTag>("mcMatch"))) {
}


void DebugZMCTruth::analyze(const Event& event, const EventSetup& setup) {
  Handle<GenParticleCollection> genParticles;
  event.getByToken(genParticlesToken_, genParticles);
  Handle<CandidateView> src;
  event.getByToken(srcToken_, src);
  cout << ">>> event has " << src->size() << " reconstructed particles in {" << src_ << "}" <<endl;
  Handle<GenParticleMatch> match;
  event.getByToken(matchToken_, match);
  cout << ">>> Z matches: ";
  for(unsigned int i = 0; i < src->size(); ++i) {
    CandidateBaseRef ref = src->refAt(i);
    GenParticleRef mc = (*match)[ref];
    cout << (mc.isNull() ? "(no)" : "(yes)");
  }
  cout << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DebugZMCTruth);

