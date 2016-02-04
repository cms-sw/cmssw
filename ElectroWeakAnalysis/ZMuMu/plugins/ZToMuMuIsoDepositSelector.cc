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

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"

using namespace reco;
using namespace isodeposit;

template<typename Isolator>
class ZToMuMuIsoDepositSelector {
public:
  ZToMuMuIsoDepositSelector(const edm::ParameterSet & cfg) :
    isolator_(cfg.template getParameter<double>("isoCut")),
    ptThreshold(cfg.getUntrackedParameter<double>("ptThreshold")),
    etEcalThreshold(cfg.getUntrackedParameter<double>("etEcalThreshold")),
    etHcalThreshold(cfg.getUntrackedParameter<double>("etHcalThreshold")),
    dRVetoTrk(cfg.getUntrackedParameter<double>("deltaRVetoTrk")),
    dRTrk(cfg.getUntrackedParameter<double>("deltaRTrk")),
    dREcal(cfg.getUntrackedParameter<double>("deltaREcal")),
    dRHcal(cfg.getUntrackedParameter<double>("deltaRHcal")),
    alpha(cfg.getUntrackedParameter<double>("alpha")),
    beta(cfg.getUntrackedParameter<double>("beta")),
    relativeIsolation(cfg.template getParameter<bool>("relativeIsolation")) {
  }

  template<typename T>
  double isolation(const T * t) const {
    const pat::IsoDeposit * trkIso = t->isoDeposit(pat::TrackIso);
    const pat::IsoDeposit * ecalIso = t->isoDeposit(pat::EcalIso);
    const pat::IsoDeposit * hcalIso = t->isoDeposit(pat::HcalIso);   
    
    Direction dir = Direction(t->eta(), t->phi());
    
    IsoDeposit::AbsVetos vetosTrk;
    vetosTrk.push_back(new ConeVeto( dir, dRVetoTrk ));
    vetosTrk.push_back(new ThresholdVeto( ptThreshold ));
    
    IsoDeposit::AbsVetos vetosEcal;
    vetosEcal.push_back(new ConeVeto( dir, 0.));
    vetosEcal.push_back(new ThresholdVeto( etEcalThreshold ));
    
    IsoDeposit::AbsVetos vetosHcal;
    vetosHcal.push_back(new ConeVeto( dir, 0. ));
    vetosHcal.push_back(new ThresholdVeto( etHcalThreshold ));

    double isovalueTrk = (trkIso->sumWithin(dRTrk,vetosTrk));
    double isovalueEcal = (ecalIso->sumWithin(dREcal,vetosEcal));
    double isovalueHcal = (hcalIso->sumWithin(dRHcal,vetosHcal));
    

    double iso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk) ;
    if(relativeIsolation) iso /= t->pt();
    return iso;
  }

  double candIsolation(const reco::Candidate* c) const {
    const pat::Muon * mu = dynamic_cast<const pat::Muon *>(c);
    if(mu != 0) return isolation(mu);
    const pat::GenericParticle * trk = dynamic_cast<const pat::GenericParticle*>(c);
    if(trk != 0) return isolation(trk);
    throw edm::Exception(edm::errors::InvalidReference) 
      << "Candidate daughter #0 is neither pat::Muons nor pat::GenericParticle\n";      
    return -1;
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
    return isolator_(candIsolation(m0), candIsolation(m1));
  }
private:
  Isolator isolator_;
  double ptThreshold,etEcalThreshold,etHcalThreshold, dRVetoTrk, dRTrk, dREcal, dRHcal, alpha, beta;  
  bool relativeIsolation;
  
};

#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

typedef SingleObjectSelector<reco::CandidateView, 
    AndSelector<ZToMuMuIsoDepositSelector<IsolatedSelector>, 
		StringCutObjectSelector<reco::Candidate> 
    > 
  > ZToMuMuIsolatedIDSelector;

typedef SingleObjectSelector<reco::CandidateView, 
    AndSelector<ZToMuMuIsoDepositSelector<NonIsolatedSelector>, 
		StringCutObjectSelector<reco::Candidate> 
    > 
  > ZToMuMuNonIsolatedIDSelector;


typedef SingleObjectSelector<reco::CandidateView, 
    AndSelector<ZToMuMuIsoDepositSelector<OneNonIsolatedSelector>, 
		StringCutObjectSelector<reco::Candidate> 
    > 
  > ZToMuMuOneNonIsolatedIDSelector;

typedef SingleObjectSelector<reco::CandidateView, 
    AndSelector<ZToMuMuIsoDepositSelector<TwoNonIsolatedSelector>, 
		StringCutObjectSelector<reco::Candidate> 
    > 
  > ZToMuMuTwoNonIsolatedIDSelector;


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZToMuMuIsolatedIDSelector);
DEFINE_FWK_MODULE(ZToMuMuNonIsolatedIDSelector);
DEFINE_FWK_MODULE(ZToMuMuOneNonIsolatedIDSelector);
DEFINE_FWK_MODULE(ZToMuMuTwoNonIsolatedIDSelector);
