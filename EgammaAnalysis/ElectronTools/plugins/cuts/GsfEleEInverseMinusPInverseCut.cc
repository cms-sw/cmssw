#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
 
class GsfEleEInverseMinusPInverseCut : public CutApplicatorBase {
public:
  GsfEleEInverseMinusPInverseCut(const edm::ParameterSet& c) :
    CutApplicatorBase(c),
    _ooemoopCutValueEB(c.getParameter<double>("eInverseMinusPInverseCutValueEB")),
    _ooemoopCutValueEE(c.getParameter<double>("eInverseMinusPInverseCutValueEE")),
    _barrelCutOff(c.getParameter<double>("barrelCutOff")){
  }
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  const double _ooemoopCutValueEB,_ooemoopCutValueEE,_barrelCutOff;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleEInverseMinusPInverseCut,
		  "GsfEleEInverseMinusPInverseCut");

CutApplicatorBase::result_type 
GsfEleEInverseMinusPInverseCut::
operator()(const reco::GsfElectronRef& cand) const{
  const float ooemoopCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _ooemoopCutValueEB : _ooemoopCutValueEE );
  const float ecal_energy_inverse = 1.0/cand->ecalEnergy();
  const float eSCoverP = cand->eSuperClusterOverP();
  return std::abs(1.0 - eSCoverP)*ecal_energy_inverse < ooemoopCutValue;
}
