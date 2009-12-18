/* \class ZToMuMuIsolationSelector
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"

struct IsolatedSelector {
  IsolatedSelector(double cut) : cut_(cut) { }
  bool operator()(double i1, double i2) const {
    return i1 < cut_ && i2 < cut_;
  }
  double cut_;
};

struct NonIsolatedSelector {
  NonIsolatedSelector(double cut) : isolated_(cut) { }
  bool operator()(double i1, double i2) const {
    return !isolated_(i1, i2);
  }
private:
  IsolatedSelector isolated_;
};

#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
namespace edm { class EventSetup; }

typedef edm::ValueMap<float> IsolationCollection;

template<typename Isolator>
class ZToMuMuIsolationSelector {
public:
  ZToMuMuIsolationSelector(const edm::ParameterSet &);
  bool operator()(const reco::Candidate&) const;
  void newEvent(const edm::Event&, const edm::EventSetup&);
  edm::InputTag  muIso1_, muIso2_;
  double isoCut_;
  Isolator isolator_;
  edm::Handle<IsolationCollection> hMuIso1_, hMuIso2_;
};

template<typename Isolator>
ZToMuMuIsolationSelector<Isolator>::ZToMuMuIsolationSelector(const edm::ParameterSet & cfg) :
  muIso1_(cfg.template getParameter<edm::InputTag>("muonIsolations1")),
  muIso2_(cfg.template getParameter<edm::InputTag>("muonIsolations2")),
  isolator_(cfg.template getParameter<double>("isoCut")) {
}

template<typename Isolator>
void ZToMuMuIsolationSelector<Isolator>::newEvent(const edm::Event& ev, const edm::EventSetup&) {
  ev.getByLabel(muIso1_, hMuIso1_);
  ev.getByLabel(muIso2_, hMuIso2_);
}

template<typename Isolator>
bool ZToMuMuIsolationSelector<Isolator>::operator()(const reco::Candidate & z) const {
  if(z.numberOfDaughters()!=2) return false;
  const reco::Candidate * dau0 = z.daughter(0);
  const reco::Candidate * dau1 = z.daughter(1);
  reco::CandidateBaseRef mu0 = dau0->masterClone();
  reco::CandidateBaseRef mu1 = dau1->masterClone();
  double iso0 = (*hMuIso1_)[mu0];
  double iso1 = (*hMuIso2_)[mu1];
  return isolator_(iso0, iso1);
}

#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"

#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
EVENTSETUP_STD_INIT_T1(ZToMuMuIsolationSelector);

typedef SingleObjectSelector<reco::CandidateView, 
    AndSelector<ZToMuMuIsolationSelector<IsolatedSelector>, 
		StringCutObjectSelector<reco::Candidate> 
    > 
  > ZToMuMuIsolatedSelector;

typedef SingleObjectSelector<reco::CandidateView, 
    AndSelector<ZToMuMuIsolationSelector<NonIsolatedSelector>, 
		StringCutObjectSelector<reco::Candidate> 
    > 
  > ZToMuMuNonIsolatedSelector;


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZToMuMuIsolatedSelector);
DEFINE_FWK_MODULE(ZToMuMuNonIsolatedSelector);
