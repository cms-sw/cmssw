/* \class ZToMuMuIsolationSelector
 *
 * \author Luca Lista, INFN
 *
 */

struct IsolatedSelector {
  IsolatedSelector(double cut) : cut_(cut) { }
  bool operator()(double i1, double i2) const {
    return i1 < cut_ && i2 < cut_;
  }
  double cut() const { return cut_; }
private:
  double cut_;
};

struct NonIsolatedSelector {
  NonIsolatedSelector(double cut) : isolated_(cut) { }
  bool operator()(double i1, double i2) const {
    return !isolated_(i1, i2);
  }
  double cut() const { return isolated_.cut(); }
private:
  IsolatedSelector isolated_;
};

struct OneNonIsolatedSelector {
  OneNonIsolatedSelector(double cut) : cut_(cut) { }
  bool operator()(double i1, double i2) const {
    return (i1 < cut_ && i2 >= cut_) || (i1 >= cut_ && i2 < cut_);
  }
  double cut() const { return cut_; }
private:
  double cut_;
};

struct TwoNonIsolatedSelector {
  TwoNonIsolatedSelector(double cut) : cut_(cut) { }
  bool operator()(double i1, double i2) const {
    return i1 >= cut_ && i2 >= cut_;
  }
  double cut() const { return cut_; }
private:
  double cut_;
};

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename Isolator>
class ZToMuMuIsolationSelector {
public:
  ZToMuMuIsolationSelector(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) :
    isolator_(cfg.template getParameter<double>("isoCut")) {
    std::string iso = cfg.template getParameter<std::string>("isolationType");
    if(iso == "track")  {
      leptonIsolation_ = & pat::Lepton<reco::Muon>::trackIso;
      trackIsolation_ = & pat::GenericParticle::trackIso;
    }
    else if(iso == "ecal") {
      leptonIsolation_ = & pat::Lepton<reco::Muon>::ecalIso;
      trackIsolation_ = & pat::GenericParticle::ecalIso;
    }
    else if(iso == "hcal") {
      leptonIsolation_ = & pat::Lepton<reco::Muon>::hcalIso;
      trackIsolation_ = & pat::GenericParticle::hcalIso;
    }
    else if(iso == "calo") {
      leptonIsolation_ = & pat::Lepton<reco::Muon>::caloIso;
      trackIsolation_ = & pat::GenericParticle::caloIso;
    }
    else   throw edm::Exception(edm::errors::Configuration)
      << "Invalid isolation type: " << iso << ". Valid types are:"
      << "'track', 'ecal', 'hcal', 'calo'\n";
  }
  bool operator()(const reco::Candidate & z) const {
    if(z.numberOfDaughters()!=2)
      throw edm::Exception(edm::errors::InvalidReference)
	<< "Candidate has " << z.numberOfDaughters() << " daughters, 2 expected\n";
    const reco::Candidate * dau0 = z.daughter(0);
    const reco::Candidate * dau1 = z.daughter(1);
    if(!(dau0->hasMasterClone()&&dau1->hasMasterClone()))
      throw edm::Exception(edm::errors::InvalidReference)
	<< "Candidate daughters have no master clone\n";
    const reco::Candidate * m0 = &*dau0->masterClone(), * m1 = &*dau1->masterClone();
    double iso0 = -1, iso1 = -1;
    const pat::Muon * mu0 = dynamic_cast<const pat::Muon *>(m0);
    if(mu0 != 0) {
      iso0 = ((*mu0).*(leptonIsolation_))();
    } else {
      const pat::GenericParticle * trk0 = dynamic_cast<const pat::GenericParticle*>(m0);
      if(trk0 != 0) {
	iso0 = ((*trk0).*(trackIsolation_))();
      } else {
	throw edm::Exception(edm::errors::InvalidReference)
	  << "Candidate daughter #0 is neither pat::Muons nor pat::GenericParticle\n";
      }
    }
    const pat::Muon * mu1 = dynamic_cast<const pat::Muon *>(m1);
    if(mu1 != 0) {
      iso1 = ((*mu1).*(leptonIsolation_))();
    } else {
      const pat::GenericParticle * trk1 = dynamic_cast<const pat::GenericParticle*>(m1);
      if(trk1 != 0) {
	iso1 = ((*trk1).*(trackIsolation_))();
      } else {
	throw edm::Exception(edm::errors::InvalidReference)
	  << "Candidate daughter #1 is neither pat::Muons nor pat::GenericParticle\n";
      }
    }
    bool pass = isolator_(iso0, iso1);
    return pass;
  }
  private:
    typedef float (pat::Lepton<reco::Muon>::*LeptonIsolationType)() const;
    typedef float (pat::GenericParticle::*TrackIsolationType)() const;
    LeptonIsolationType leptonIsolation_;
    TrackIsolationType trackIsolation_;
    Isolator isolator_;
};

namespace dummy {
  void Isolationdummy() {
    pat::Lepton<reco::Muon> pat;
    //ignore return values
    pat.trackIso(); pat.ecalIso(); pat.hcalIso(); pat.caloIso();
  }
}

#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

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


typedef SingleObjectSelector<reco::CandidateView,
    AndSelector<ZToMuMuIsolationSelector<OneNonIsolatedSelector>,
		StringCutObjectSelector<reco::Candidate>
    >
  > ZToMuMuOneNonIsolatedSelector;

typedef SingleObjectSelector<reco::CandidateView,
    AndSelector<ZToMuMuIsolationSelector<TwoNonIsolatedSelector>,
		StringCutObjectSelector<reco::Candidate>
    >
  > ZToMuMuTwoNonIsolatedSelector;


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZToMuMuIsolatedSelector);
DEFINE_FWK_MODULE(ZToMuMuNonIsolatedSelector);
DEFINE_FWK_MODULE(ZToMuMuOneNonIsolatedSelector);
DEFINE_FWK_MODULE(ZToMuMuTwoNonIsolatedSelector);
