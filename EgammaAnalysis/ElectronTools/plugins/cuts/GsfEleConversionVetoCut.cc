#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

class GsfEleConversionVetoCut : public CutApplicatorBase {
public:
  GsfEleConversionVetoCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c) {    
  }
  
  result_type operator()(const reco::GsfElectron&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:  
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleConversionVetoCut,
		  "GsfEleConversionVetoCut");

CutApplicatorBase::result_type 
GsfEleConversionVetoCut::
operator()(const reco::GsfElectron& cand) const{
  return false;
  /*
  return !ConversionTools::hasMatchedConversion(cand, 
						conversions, 
						beamspot.position());
  */
}
