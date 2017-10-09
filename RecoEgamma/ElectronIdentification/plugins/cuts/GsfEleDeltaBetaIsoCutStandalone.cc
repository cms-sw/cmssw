#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"


class GsfEleDeltaBetaIsoCutStandalone : public CutApplicatorBase {
public:
  GsfEleDeltaBetaIsoCutStandalone(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;
  
  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const float _isoCutEBLowPt,_isoCutEBHighPt,_isoCutEELowPt,_isoCutEEHighPt;
  const float _deltaBetaConstant,_ptCutOff,_barrelCutOff;
  const bool _relativeIso;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDeltaBetaIsoCutStandalone,
		  "GsfEleDeltaBetaIsoCutStandalone");

GsfEleDeltaBetaIsoCutStandalone::GsfEleDeltaBetaIsoCutStandalone(const edm::ParameterSet& c) :
  CutApplicatorBase(c),
  _isoCutEBLowPt(c.getParameter<double>("isoCutEBLowPt")),
  _isoCutEBHighPt(c.getParameter<double>("isoCutEBHighPt")),
  _isoCutEELowPt(c.getParameter<double>("isoCutEELowPt")),
  _isoCutEEHighPt(c.getParameter<double>("isoCutEEHighPt")),
  _deltaBetaConstant(c.getParameter<double>("deltaBetaConstant")),
  _ptCutOff(c.getParameter<double>("ptCutOff")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _relativeIso(c.getParameter<bool>("isRelativeIso")) {  
}

CutApplicatorBase::result_type 
GsfEleDeltaBetaIsoCutStandalone::
operator()(const reco::GsfElectronPtr& cand) const{
  const float isoCut = 
    ( cand->p4().pt() < _ptCutOff ? 
      ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ?
	_isoCutEBLowPt : _isoCutEELowPt ) :
      ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ?
	_isoCutEBHighPt : _isoCutEEHighPt ) );
  const reco::GsfElectron::PflowIsolationVariables& pfIso = 
    cand->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho = pfIso.sumPhotonEt;
  const float puchad = pfIso.sumPUPt;
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso < isoCut;
}

double GsfEleDeltaBetaIsoCutStandalone::
value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  const reco::GsfElectron::PflowIsolationVariables& pfIso = 
    ele->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho = pfIso.sumPhotonEt;
  const float puchad = pfIso.sumPUPt;
  float iso = chad + std::max(0.0f, nhad + pho - _deltaBetaConstant*puchad);
  if( _relativeIso ) iso /= cand->p4().pt();
  return iso;
}
