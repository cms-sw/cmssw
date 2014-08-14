#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "EgammaAnalysis/ElectronTools/interface/EBEECutValues.h"

class GsfEleTrkPtIsoCut : public CutApplicatorBase {
public:
  GsfEleTrkPtIsoCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
 
  EBEECutValues slopeTerm_;
  EBEECutValues slopeStart_;
  EBEECutValues constTerm_;
  
  
  edm::Handle<double> rhoHandle_;
  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleTrkPtIsoCut,
		  "GsfEleTrkPtIsoCut");

GsfEleTrkPtIsoCut::GsfEleTrkPtIsoCut(const edm::ParameterSet& params) :
  CutApplicatorBase(params),
  slopeTerm_(params,"slopeTerm"),
  slopeStart_(params,"slopeStart"),
  constTerm_(params,"constTerm")
{

}


CutApplicatorBase::result_type 
GsfEleTrkPtIsoCut::
operator()(const reco::GsfElectronRef& cand) const{  
  
  const float isolTrkPt = cand->dr03TkSumPt();

  const float et = cand->et();
  const float cutValue = et > slopeStart_(cand)  ? slopeTerm_(cand)*(et-slopeStart_(cand)) + constTerm_(cand) : constTerm_(cand);
  return isolTrkPt < cutValue;
}
