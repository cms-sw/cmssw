#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleMissingHitsCut : public CutApplicatorBase {
public:
  GsfEleMissingHitsCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _maxMissingHitsEB(c.getParameter<unsigned>("maxMissingHitsEB")),
    _maxMissingHitsEE(c.getParameter<unsigned>("maxMissingHitsEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")){
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const unsigned _maxMissingHitsEB, _maxMissingHitsEE;
  const double _barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleMissingHitsCut,
		  "GsfEleMissingHitsCut");

CutApplicatorBase::result_type 
GsfEleMissingHitsCut::
operator()(const reco::GsfElectronPtr& cand) const{ 
  constexpr reco::HitPattern::HitCategory missingHitType =
    reco::HitPattern::MISSING_INNER_HITS;
  const unsigned maxMissingHits = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _maxMissingHitsEB : _maxMissingHitsEE );
  const unsigned mHits = 
    cand->gsfTrack()->hitPattern().numberOfHits(missingHitType);
  return mHits <= maxMissingHits;
}

double GsfEleMissingHitsCut::value(const reco::CandidatePtr& cand) const {
  constexpr reco::HitPattern::HitCategory missingHitType =
    reco::HitPattern::MISSING_INNER_HITS;
  reco::GsfElectronPtr ele(cand);
  const unsigned mHits = 
    ele->gsfTrack()->hitPattern().numberOfHits(missingHitType);
  return mHits;
}
