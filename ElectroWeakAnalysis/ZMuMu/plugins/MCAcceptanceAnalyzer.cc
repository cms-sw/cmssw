#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
  unsigned int n = c->numberOfDaughters();
  for(unsigned int i = 0; i < n; ++i) {
    const Candidate * d = c->daughter(i);
    if(fabs(d->pdgId())==13) return d;
  }
  return 0;
}

struct ZSelector {                  // modify this selector in order to return an integer (0: no eta cut, 1: eta cut only, 2 eta && pt cut, 3: eta, pt and Mass cut, 4: mass cut on the denominator Z MC)
  ZSelector(double ptMin, double etaDau0Min, double etaDau0Max,double etaDau1Min, double etaDau1Max, double massMin, double massMax, double massMinZMC, double massMaxZMC) :
    ptMin_(ptMin), etaDau0Min_(etaDau0Min),etaDau0Max_(etaDau0Max),  etaDau1Min_(etaDau1Min),etaDau1Max_(etaDau1Max),
    massMin_(massMin), massMax_(massMax), massMinZMC_(massMinZMC), massMaxZMC_(massMaxZMC) { }
  int operator()(const Candidate& c) const {
    //   std::cout << "c.numberOfDaughters(): " << c.numberOfDaughters()<< std::endl;
    if (c.numberOfDaughters()<2) return 0;
    if (c.numberOfDaughters()>=6)      return 0;
    const Candidate * d0 = c.daughter(0);
    const Candidate * d1 = c.daughter(1);
    if(c.numberOfDaughters()>2) {
    if (d0->numberOfDaughters()>0)  d0 = mcMuDaughter(d0);
    if (d1->numberOfDaughters()>0)  d1 = mcMuDaughter(d1);
    }
    int temp_cut= 0;
    /// allowing asymmetric cut....
    if( ( fabs(d0->eta()) > etaDau0Min_ && fabs(d1->eta())>etaDau1Min_ && fabs(d0->eta()) < etaDau0Max_ && fabs(d1->eta()) <etaDau1Max_ ) ||  ( fabs(d0->eta()) > etaDau1Min_ && fabs(d1->eta())>etaDau0Min_ && fabs(d0->eta()) < etaDau1Max_ && fabs(d1->eta()) <etaDau0Max_ ) ) {
      temp_cut=1;
      if(d0->pt() > ptMin_ && d1->pt() > ptMin_) {
	temp_cut=2;
	double m = (d0->p4() + d1->p4()).mass();
	if(m > massMin_ && m < massMax_) temp_cut=3;
        if (c.mass()> massMinZMC_ && c.mass() < massMaxZMC_) temp_cut =4;
      }
    }

    return temp_cut;
  }
  double ptMin_, etaDau0Min_, etaDau0Max_, etaDau1Min_, etaDau1Max_,  massMin_, massMax_, massMinZMC_, massMaxZMC_;
};

class MCAcceptanceAnalyzer : public EDAnalyzer {
public:
  MCAcceptanceAnalyzer(const ParameterSet& cfg);
private:
  void analyze(const Event&, const EventSetup&) override;
  void endJob() override;
  InputTag zToMuMu_, zToMuMuMC_, zToMuMuMatched_;
  EDGetTokenT<CandidateView> zToMuMuToken_;
  EDGetTokenT<CandidateView> zToMuMuMCToken_;
  EDGetTokenT<GenParticleMatch > zToMuMuMatchedToken_;
  long nZToMuMu_, selZToMuMu_, nZToMuMuMC_, selZToMuMuMC_, nZToMuMuMCMatched_, selZToMuMuMCMatched_, nZToMuMuMCDen_;
  ZSelector select_, select_OnlyMassCut_;
};

MCAcceptanceAnalyzer::MCAcceptanceAnalyzer(const ParameterSet& cfg) :
  zToMuMu_(cfg.getParameter<InputTag>("zToMuMu")),
  zToMuMuMC_(cfg.getParameter<InputTag>("zToMuMuMC")),
  zToMuMuMatched_(cfg.getParameter<InputTag>("zToMuMuMatched")),
  zToMuMuToken_(consumes<CandidateView>(zToMuMu_)),
  zToMuMuMCToken_(consumes<CandidateView>(zToMuMuMC_)),
  zToMuMuMatchedToken_(consumes<GenParticleMatch> (zToMuMuMatched_)),
  nZToMuMu_(0), selZToMuMu_(0),
  nZToMuMuMC_(0), selZToMuMuMC_(0),
  nZToMuMuMCMatched_(0), selZToMuMuMCMatched_(0), nZToMuMuMCDen_(0),
  select_(cfg.getParameter<double>("ptMin"), cfg.getParameter<double>("etaDau0Min"), cfg.getParameter<double>("etaDau0Max"), cfg.getParameter<double>("etaDau1Min"), cfg.getParameter<double>("etaDau1Max"),cfg.getParameter<double>("massMin"), cfg.getParameter<double>("massMax"), cfg.getParameter<double>("massMinZMC"), cfg.getParameter<double>("massMaxZMC")  ),
  select_OnlyMassCut_(-1, -9999,9999 , -9999, 9999,0, 0, cfg.getParameter<double>("massMinZMC"), cfg.getParameter<double>("massMaxZMC")  )
{
}

void MCAcceptanceAnalyzer::analyze(const Event& evt, const EventSetup&) {
  Handle<CandidateView> zToMuMu;
  evt.getByToken(zToMuMuToken_, zToMuMu);
  Handle<CandidateView> zToMuMuMC;
  evt.getByToken(zToMuMuMCToken_, zToMuMuMC);
  Handle<GenParticleMatch > zToMuMuMatched;
  evt.getByToken(zToMuMuMatchedToken_, zToMuMuMatched);
  //  long nZToMuMu = zToMuMu->size();
  long nZToMuMuMC = zToMuMuMC->size();
  long nZToMuMuMatched = zToMuMuMatched->size();


  // cout << ">>> " << zToMuMu_ << " has " << nZToMuMu << " entries" << endl;
  //cout << ">>> " << zToMuMuMC_ << " has " << nZToMuMuMC << " entries" << endl;
  //cout << ">>> " << zToMuMuMatched_ << " has " << nZToMuMuMatched << " entries" << endl;



  nZToMuMuMC_ += nZToMuMuMC;
  for(long i = 0; i < nZToMuMuMC; ++i) {
    const Candidate & z = (*zToMuMuMC)[i];
    if(select_(z)==4) ++selZToMuMuMC_;
    if(select_OnlyMassCut_(z)==4) ++nZToMuMuMCDen_;

  }


  for(long i = 0; i < nZToMuMuMatched; ++i) {

    const Candidate & z = (*zToMuMu)[i];
    CandidateBaseRef zRef = zToMuMu->refAt(i);
    GenParticleRef mcRef = (*zToMuMuMatched)[zRef];


    if(mcRef.isNonnull()) {   // z candidate matched to Z MC
    ++nZToMuMu_;
    ++nZToMuMuMCMatched_;

    int selectZ = select_(z);
    if(selectZ==4) ++selZToMuMu_;



      int selectMC = select_(*mcRef);

      if(selectMC==4) ++selZToMuMuMCMatched_;

      if(selectZ != selectMC) {
	cout << ">>> select reco: " << selectZ << ", select mc: " << selectMC << endl;
        if ((selectZ * selectMC) ==0 ) break;
	if (z.numberOfDaughters()> 1){
	const Candidate * d0 = z.daughter(0), * d1 = z.daughter(1);
	if (mcRef->numberOfDaughters()>1){
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
      // to avoid double counting
      if ((selectZ==3) && (selectMC==3)) break;
    }
   }

}

void MCAcceptanceAnalyzer::endJob() {
  double effZToMuMu = double(selZToMuMu_)/double(nZToMuMu_);
  double errZToMuMu = sqrt(effZToMuMu*(1. - effZToMuMu)/nZToMuMu_);
  double effZToMuMuMC = double(selZToMuMuMC_)/double(nZToMuMuMC_);
  double errZToMuMuMC = sqrt(effZToMuMuMC*(1. - effZToMuMuMC)/nZToMuMuMC_);
  double effZToMuMuMCDen = double(selZToMuMuMC_)/double(nZToMuMuMCDen_);
  double errZToMuMuMCDen = sqrt(effZToMuMuMCDen*(1. - effZToMuMuMCDen)/nZToMuMuMCDen_);
  double effZToMuMuMCMatched = double(selZToMuMuMCMatched_)/double(nZToMuMuMCMatched_);
  double errZToMuMuMCMatched = sqrt(effZToMuMuMCMatched*(1. - effZToMuMuMCMatched)/nZToMuMuMCMatched_);
  cout << ">>> " << zToMuMu_ << ": " << selZToMuMu_ << "/" << nZToMuMu_
       << " = " << effZToMuMu << " +/- " << errZToMuMu << endl;
  cout << ">>> " << zToMuMuMC_ << " - matched: " << selZToMuMuMCMatched_ << "/" << nZToMuMuMCMatched_
       << " = " << effZToMuMuMCMatched << " +/- " << errZToMuMuMCMatched << endl;
  cout << " if the two numbers above are the same we can neglete resolution effect and quote the acceptance as the number below.... " << endl;
  cout << "********* acceptance m>sampleMCMassCut (usually 20 or 40) ********  " << endl;
  cout << ">>> " << zToMuMuMC_ << ": " << selZToMuMuMC_ << "/" << nZToMuMuMC_
       << " = " << effZToMuMuMC << " +/- " << errZToMuMuMC << endl;
  cout << "********* acceptance in the given mass range ********  " << endl;
  cout << ">>> " << zToMuMuMC_ << ": " << selZToMuMuMC_ << "/" << nZToMuMuMCDen_
       << " = " << effZToMuMuMCDen << " +/- " << errZToMuMuMCDen << endl;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCAcceptanceAnalyzer);

