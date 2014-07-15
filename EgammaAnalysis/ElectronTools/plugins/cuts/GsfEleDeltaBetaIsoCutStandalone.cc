#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"


class GsfEleDeltaBetaIsoCutStandalone : public CutApplicatorBase {
public:
  GsfEleDeltaBetaIsoCutStandalone(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronRef&) const override final;
  
  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  float _deltaBetaConstant;
  bool _relativeIso;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDeltaBetaIsoCutStandalone,
		  "GsfEleDeltaBetaIsoCutStandalone");

GsfEleDeltaBetaIsoCutStandalone::GsfEleDeltaBetaIsoCutStandalone(const edm::ParameterSet& c) :
  CutApplicatorBase(c),
  _deltaBetaConstant(c.getParameter<double>("deltaBetaConstant")),
  _relativeIso(c.getParameter<bool>("isRelativeIso")) {  
}

CutApplicatorBase::result_type 
GsfEleDeltaBetaIsoCutStandalone::
operator()(const reco::GsfElectronRef& cand) const{  
  const reco::GsfElectron::PflowIsolationVariables& pfIso = 
    cand->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho = pfIso.sumPhotonEt;
  const float puchad = pfIso.sumPUPt;
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso;
}
