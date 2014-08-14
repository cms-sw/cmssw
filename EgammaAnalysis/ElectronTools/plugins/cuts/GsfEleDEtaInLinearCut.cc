#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "EgammaAnalysis/ElectronTools/interface/EBEECutValues.h"

class GsfEleDEtaInLinearCut : public CutApplicatorBase {
public:
  GsfEleDEtaInLinearCut(const edm::ParameterSet& param) :
    CutApplicatorBase(param),
    slopeTerm_(param,"slopeTerm"),
    constTerm_(param,"constTerm"),
    minValue_(param,"minValue")
  {
  }
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const EBEECutValues slopeTerm_;
  const EBEECutValues constTerm_;
  const EBEECutValues minValue_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDEtaInLinearCut,
		  "GsfEleDEtaInLinearCut");

CutApplicatorBase::result_type 
GsfEleDEtaInLinearCut::
operator()(const reco::GsfElectronRef& cand) const
{  
  double cutValue = std::max(constTerm_(cand)+slopeTerm_(cand)*cand->et(),minValue_(cand));
  return std::abs(cand->deltaEtaSuperClusterTrackAtVtx())<cutValue;
 
}
