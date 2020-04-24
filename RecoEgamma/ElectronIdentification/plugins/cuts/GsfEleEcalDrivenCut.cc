#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleEcalDrivenCut : public CutApplicatorBase {
public:
  GsfEleEcalDrivenCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
    return ELECTRON; 
  }

private:
  static bool isValidCutVal(int val);

private:
  enum EcalDrivenCode{IGNORE=-1,FAIL=0,PASS=1};
  const int ecalDrivenEB_, ecalDrivenEE_;
  const double barrelCutOff_;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleEcalDrivenCut,
		  "GsfEleEcalDrivenCut");


 
GsfEleEcalDrivenCut::GsfEleEcalDrivenCut(const edm::ParameterSet& c) :
  CutApplicatorBase(c),
  ecalDrivenEB_(static_cast<EcalDrivenCode>(c.getParameter<int>("ecalDrivenEB"))),
  ecalDrivenEE_(static_cast<EcalDrivenCode>(c.getParameter<int>("ecalDrivenEE"))),
  barrelCutOff_(c.getParameter<double>("barrelCutOff"))
{
  if(!isValidCutVal(ecalDrivenEB_) || !isValidCutVal(ecalDrivenEE_)){
    throw edm::Exception(edm::errors::Configuration)
      <<"error in constructing GsfEleEcalDrivenCut"<<std::endl
      <<"values of ecalDrivenEB: "<<ecalDrivenEB_<<" and/or ecalDrivenEE: "<<ecalDrivenEE_<<" are invalid "<<std::endl
      <<"allowed values are IGNORE:"<<IGNORE<<" FAIL:"<<FAIL<<" PASS:"<<PASS;
  }
}

CutApplicatorBase::result_type 
GsfEleEcalDrivenCut::
operator()(const reco::GsfElectronPtr& cand) const{ 
  const auto ecalDrivenRequirement =  std::abs(cand->superCluster()->position().eta()) < barrelCutOff_ ? 
    ecalDrivenEB_ : ecalDrivenEE_;
  if(ecalDrivenRequirement==IGNORE) return true;
  else if(ecalDrivenRequirement==FAIL) return !cand->ecalDriven();
  else if(ecalDrivenRequirement==PASS) return cand->ecalDriven();
  else{  
    throw edm::Exception(edm::errors::LogicError)
      <<"error in "<<__FILE__<<" line "<<__LINE__<<std::endl
      <<"default option should not be reached, code has been updated without changing the logic, this needs to be fixed";
  }
}

double GsfEleEcalDrivenCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return ele->ecalDriven();
}

bool GsfEleEcalDrivenCut::isValidCutVal(int val)
{
  if(val==IGNORE) return true;
  if(val==FAIL) return true;
  if(val==PASS) return true;
  return false;
}
