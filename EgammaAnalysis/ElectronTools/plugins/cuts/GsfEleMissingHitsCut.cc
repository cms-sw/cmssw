#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleMissingHitsCut : public CutApplicatorBase {
public:
  GsfEleMissingHitsCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _maxMissingHits(c.getParameter<unsigned>("maxMissingHits")) {
  }
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const unsigned _maxMissingHits;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleMissingHitsCut,
		  "GsfEleMissingHitsCut");

CutApplicatorBase::result_type 
GsfEleMissingHitsCut::
operator()(const reco::GsfElectronRef& cand) const{ 
  const unsigned mHits = 
    cand->gsfTrack()->trackerExpectedHitsInner().numberOfHits();
  return mHits <= _maxMissingHits;
}
