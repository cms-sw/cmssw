#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleEcalDrivenCut : public CutApplicatorBase {
public:
  GsfEleEcalDrivenCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    ecalDrivenEB_(c.getParameter<int>("ecalDrivenEB")),
    ecalDrivenEE_(c.getParameter<int>("ecalDrivenEE")),
    barrelCutOff_(c.getParameter<double>("barrelCutOff")){
  }
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const int ecalDrivenEB_, ecalDrivenEE_;// -1 ignore, 0 = fail ecalDriven, 1 =pass ecalDriven
  const double barrelCutOff_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleEcalDrivenCut,
		  "GsfEleEcalDrivenCut");

CutApplicatorBase::result_type 
GsfEleEcalDrivenCut::
operator()(const reco::GsfElectronPtr& cand) const{ 
  const bool ecalDriven =  std::abs(cand->superCluster()->position().eta()) < barrelCutOff_ ? 
    ecalDrivenEB_ : ecalDrivenEE_;
  if(ecalDriven<0) return true;
  else if(ecalDriven==0) return !cand->ecalDriven();
  else return cand->ecalDriven();
}

double GsfEleEcalDrivenCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return ele->ecalDriven();
}
