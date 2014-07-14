#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
 
class GsfEleEInverseMinusPInverseCut : public CutApplicatorBase {
public:
  GsfEleEInverseMinusPInverseCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _ooemoopCutValue(c.getParameter<double>("eInverseMinusPInverseCutValue")) {
  }
  
  result_type operator()(const reco::GsfElectron&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _ooemoopCutValue;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleEInverseMinusPInverseCut,
		  "GsfEleEInverseMinusPInverseCut");

CutApplicatorBase::result_type 
GsfEleEInverseMinusPInverseCut::
operator()(const reco::GsfElectron& cand) const{
  const float ecal_energy_inverse = 1.0/cand.ecalEnergy();
  const float eSCoverP = cand.eSuperClusterOverP();
  return std::abs(1.0 - eSCoverP)*ecal_energy_inverse < _ooemoopCutValue;
}
