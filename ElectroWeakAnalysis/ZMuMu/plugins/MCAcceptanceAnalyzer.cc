#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <iostream>

using namespace edm;
using namespace reco;
using namespace std;

const Candidate * mcMuDaughter(const Candidate * c) {
  size_t n = c->numberOfDaughters();
  for(size_t i = 0; i < n; ++i) {
    const Candidate * d = c->daughter(i);
    if(fabs(d->pdgId())==13) return d;
  }
  return 0;
}

struct ZSelector {
  ZSelector(double ptMin, double etaMax, double massMin, double massMax) :
    ptMin_(ptMin), etaMax_(etaMax), 
    massMin_(massMin), massMax_(massMax) { }
  bool operator()(const Candidate& c) const {
    const Candidate * d0 = c.daughter(0);
    const Candidate * d1 = c.daughter(1);
    if(c.numberOfDaughters()>2) {
      d0 = mcMuDaughter(d0);
      d1 = mcMuDaughter(d1);
    }
    if(d0->pt() < ptMin_ || fabs(d0->eta()) >etaMax_) return false; 
    if(d1->pt() < ptMin_ || fabs(d1->eta()) >etaMax_) return false; 
    double m = (d0->p4() + d1->p4()).mass();
    if(m < massMin_ || m > massMax_) return false;
    return true;
  }
  double ptMin_, etaMax_, massMin_, massMax_;
};

class MCAcceptanceAnalyzer : public EDAnalyzer {
public:
  MCAcceptanceAnalyzer(const ParameterSet& cfg);
private:
  void analyze(const Event&, const EventSetup&);
  void endJob();
  InputTag zToMuMu_, zToMuMuMC_, mcMap_;
  long nZToMuMu_, selZToMuMu_, nZToMuMuMC_, selZToMuMuMC_, nZToMuMuMCMatched_, selZToMuMuMCMatched_;
  ZSelector select_;
};

MCAcceptanceAnalyzer::MCAcceptanceAnalyzer(const ParameterSet& cfg) :
  zToMuMu_(cfg.getParameter<InputTag>("zToMuMu")),
  zToMuMuMC_(cfg.getParameter<InputTag>("zToMuMuMC")),
  mcMap_(cfg.getParameter<InputTag>("mcMap")),
  nZToMuMu_(0), selZToMuMu_(0), 
  nZToMuMuMC_(0), selZToMuMuMC_(0),
  nZToMuMuMCMatched_(0), selZToMuMuMCMatched_(0),
  select_(cfg.getParameter<double>("ptMin"), cfg.getParameter<double>("etaMax"),
	  cfg.getParameter<double>("massMin"), cfg.getParameter<double>("massMax")) {
}

void MCAcceptanceAnalyzer::analyze(const Event& evt, const EventSetup&) {
  Handle<CandidateView> zToMuMu;
  evt.getByLabel(zToMuMu_, zToMuMu);
  Handle<CandidateView> zToMuMuMC;
  evt.getByLabel(zToMuMuMC_, zToMuMuMC);
  Handle<GenParticleMatch> mcMap;
  evt.getByLabel(mcMap_, mcMap);
  long nZToMuMu = zToMuMu->size();
  long nZToMuMuMC = zToMuMuMC->size();
  cout << ">>> " << zToMuMu_ << " has " << nZToMuMu << " entries" << endl;   
  cout << ">>> " << zToMuMuMC_ << " has " << nZToMuMuMC << " entries" << endl;   
  nZToMuMuMC_ += nZToMuMuMC;
  for(long i = 0; i < nZToMuMuMC; ++i) { 
    const Candidate & z = (*zToMuMuMC)[i];
    if(select_(z)) ++selZToMuMuMC_;
  }
  for(long i = 0; i < nZToMuMu; ++i) { 
    const Candidate & z = (*zToMuMu)[i];
    CandidateBaseRef zRef = zToMuMu->refAt(i);
    GenParticleRef mcRef = (*mcMap)[zRef];
    if(mcRef.isNonnull()) {
      ++nZToMuMu_;
      ++nZToMuMuMCMatched_;
      bool selectZ = select_(z), selectMC = select_(*mcRef);
      if(selectZ) ++selZToMuMu_;
      if(selectMC) ++selZToMuMuMCMatched_;
      if(selectZ != selectMC) {
	cout << ">>> select reco: " << selectZ << ", select mc: " << selectMC << endl;
	const Candidate * d0 = z.daughter(0), * d1 = z.daughter(1);
	const Candidate * mcd0 = mcMuDaughter(mcRef->daughter(0)),
	  * mcd1 = mcMuDaughter(mcRef->daughter(1));
	double m = z.mass(), mcm = (mcd0->p4()+mcd1->p4()).mass();
	cout << ">>> reco pt1, eta1: " << d0->pt() <<", " << d0->eta() 
	     << ", 2: " << d1->pt() << ", " << d1->eta()
	     << ", mass = " << m << endl; 
	cout << ">>> mc   pt1, eta1: " << mcd0->pt() <<", " << mcd0->eta()
	     << ", 2: " << mcd1->pt() << ", " << mcd1->eta()
	     << ", mass = " << mcm << endl; 
     }
    }
  }
}

void MCAcceptanceAnalyzer::endJob() {
  double effZToMuMu = double(selZToMuMu_)/double(nZToMuMu_);
  double errZToMuMu = sqrt(effZToMuMu*(1. - effZToMuMu)/nZToMuMu_);
  double effZToMuMuMC = double(selZToMuMuMC_)/double(nZToMuMuMC_);
  double errZToMuMuMC = sqrt(effZToMuMuMC*(1. - effZToMuMuMC)/nZToMuMuMC_);
  double effZToMuMuMCMatched = double(selZToMuMuMCMatched_)/double(nZToMuMuMCMatched_);
  double errZToMuMuMCMatched = sqrt(effZToMuMuMCMatched*(1. - effZToMuMuMCMatched)/nZToMuMuMCMatched_);
  cout << ">>> " << zToMuMu_ << ": " << selZToMuMu_ << "/" << nZToMuMu_ 
       << " = " << effZToMuMu << " +/- " << errZToMuMu << endl;
  cout << ">>> " << zToMuMuMC_ << " - matched: " << selZToMuMuMCMatched_ << "/" << nZToMuMuMCMatched_ 
       << " = " << effZToMuMuMCMatched << " +/- " << errZToMuMuMCMatched << endl;
  cout << ">>> " << zToMuMuMC_ << ": " << selZToMuMuMC_ << "/" << nZToMuMuMC_ 
       << " = " << effZToMuMuMC << " +/- " << errZToMuMuMC << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCAcceptanceAnalyzer);

