#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <iostream>

class DimuonStatistics : public edm::EDAnalyzer {
public:
  DimuonStatistics(const edm::ParameterSet & cfg);
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
private:
  edm::InputTag src_;
  edm::EDGetTokenT<reco::CandidateView> srcToken_;
  std::vector<unsigned int> matched_, unMatched_;
  double  ptMin_, massMin_,massMax_,  etaMin_,  etaMax_,  trkIso_;
};

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
using namespace std;
using namespace reco;
using namespace edm;
const unsigned int maxEntries = 10;

DimuonStatistics::DimuonStatistics(const edm::ParameterSet & cfg) :
  src_(cfg.getParameter<InputTag>("src")),
  srcToken_(consumes<CandidateView>(src_)),
  matched_(maxEntries + 1, 0),
  unMatched_(maxEntries + 1, 0),
  ptMin_(cfg.getUntrackedParameter<double>("ptMin")),
  massMin_(cfg.getUntrackedParameter<double>("massMin")),
  massMax_(cfg.getUntrackedParameter<double>("massMax")),
  etaMin_(cfg.getUntrackedParameter<double>("etaMin")),
  etaMax_(cfg.getUntrackedParameter<double>("etaMax")),
  trkIso_(cfg.getUntrackedParameter<double>("trkIso")){
}

void DimuonStatistics::endJob() {
  cout << src_.encode() << endl ;
  cout << " == Matched == " << endl;
  for(unsigned int i = 0; i <= maxEntries; ++i) {
    cout << i << ": " << matched_[i];
    if (i < maxEntries) cout << ", ";
  }
  cout << endl;
  cout << " == unMatched == " << endl;
  for(unsigned int i = 0; i <= maxEntries; ++i) {
    cout << i << ": " << unMatched_[i];
    if (i < maxEntries) cout << ", ";
}
  cout << endl;
}

void DimuonStatistics::analyze(const edm::Event& evt, const edm::EventSetup&) {
  Handle<CandidateView> src;
  evt.getByToken(srcToken_, src);
  double trackIso1=-1;
  double trackIso2=-1;
  int j=0;
  unsigned int matched = 0, unMatched = 0;
  cout << ">> entries in " << src_ << ": " << src->size() << endl;
  for(CandidateView::const_iterator i = src->begin(); i != src->end(); ++i) {
    j++;
    const Candidate * dau1 = i->daughter(0);
    const Candidate * dau2 = i->daughter(1);
    if(dau1 == 0|| dau2 == 0)
      throw Exception(errors::InvalidReference) <<
	"one of the two daughter does not exist\n";
    const Candidate * c1 = dau1->masterClone().get();
    GenParticleRef mc1;
    const pat::Muon * mu1 = dynamic_cast<const pat::Muon*>(c1);
    if(mu1 != 0) {
      mc1 = mu1->genParticleRef();
      //     if (mc1.isNonnull()) cout << "DimuonStatistics> genParticleRef1 " << mc1->pdgId() << endl;
      trackIso1=mu1->trackIso();
    } else {
      const pat::GenericParticle * gp1 = dynamic_cast<const pat::GenericParticle*>(c1);
      if(gp1 == 0)
	throw Exception(errors::InvalidReference) <<
	  "first of two daughter is neither a pat::Muon not pat::GenericParticle\n";
      mc1 = gp1->genParticleRef();
    }
    const Candidate * c2 = dau2->masterClone().get();
    GenParticleRef mc2;
    const pat::Muon * mu2 = dynamic_cast<const pat::Muon*>(c2);
    if(mu2 != 0) {
      mc2 = mu2->genParticleRef();
      //      if (mc2.isNonnull()) cout << "DimuonStatistics> genParticleRef2 " << mc2->pdgId() << endl;
     trackIso2=mu2->trackIso();
    } else {
      const pat::GenericParticle * gp2 = dynamic_cast<const pat::GenericParticle*>(c2);
      if(gp2 == 0)
	throw Exception(errors::InvalidReference) <<
	  "first of two daughter is neither a pat::Muon not pat::GenericParticle\n";
      mc2 = gp2->genParticleRef();
    }
    GenParticleRef dimuonMatch;
    if(mc1.isNonnull() && mc2.isNonnull()) {
      cout << "DimuonStatistics> mc1: " << mc1->pdgId() << ", mc2: " << mc2->pdgId() << endl;
      int k=0;
      do {
	k++;
	mc1 = mc1->numberOfMothers() > 0 ? mc1->motherRef() : GenParticleRef();
	mc2 = mc2->numberOfMothers() > 0 ? mc2->motherRef() : GenParticleRef();
	//	cout << "DimuonStatistics> do loop: " << k << "  id1 " << mc1->pdgId() << "    id2 " << mc2->pdgId() << endl;
      } while(mc1 != mc2 && mc1.isNonnull() && mc2.isNonnull());
      if(mc1.isNonnull() && mc2.isNonnull() && mc1->pdgId()==23) {
	dimuonMatch = mc1;
      }
    }
    //    cout << "DimuonMatcher> dimuonMatch " << dimuonMatch.isNonnull() << endl;
    if( (fabs(dau1->eta()) > etaMin_  && fabs(dau1->eta()) < etaMax_)   && dau1->pt()>ptMin_ &&
     ( (fabs(dau2->eta()) > etaMin_ ) && (fabs(dau2->eta()) < etaMax_) ) && dau2->pt()>ptMin_ &&
  trackIso1 < trkIso_ &&  trackIso2 < trkIso_ &&  		      	   i->mass() > massMin_ && i->mass() < massMax_
	)  {
      cout << "dimuon mass " << i->mass() << endl;
      if(dimuonMatch.isNonnull())
	++matched;
      else
	++unMatched;


    }
  }
  cout << "matched: " << matched << ", unmatched: " << unMatched << endl;

  if(matched > maxEntries) matched = maxEntries;
  if(unMatched > maxEntries) unMatched = maxEntries;
  ++matched_[matched];
  ++unMatched_[unMatched];
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DimuonStatistics);
