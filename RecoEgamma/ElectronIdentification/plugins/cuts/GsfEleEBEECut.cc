#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class GsfEleEBEECut : public CutApplicatorBase {
public:
  GsfEleEBEECut(const edm::ParameterSet& c)
      : CutApplicatorBase(c), cutFormula_(c.getParameter<std::string>("cutString")), cutValue_(c, "cutValue") {}

  result_type operator()(const reco::GsfElectronPtr& cand) const final { return cutFormula_(*cand) < cutValue_(cand); }

  double value(const reco::CandidatePtr& cand) const final {
    reco::GsfElectronPtr ele(cand);
    return cutFormula_(*ele);
  }

  CandidateType candidateType() const final { return ELECTRON; }

private:
  StringObjectFunction<reco::GsfElectron> cutFormula_;
  const EBEECutValuesT<double> cutValue_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleEBEECut, "GsfEleEBEECut");
