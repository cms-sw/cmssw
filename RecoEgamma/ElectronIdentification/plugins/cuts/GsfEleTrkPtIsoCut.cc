#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "RecoEgamma/EgammaTools/interface/EBEECutValues.h"

class GsfEleTrkPtIsoCut : public CutApplicatorBase {
public:
  GsfEleTrkPtIsoCut(const edm::ParameterSet& c);

  result_type operator()(const reco::GsfElectronPtr&) const final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { return ELECTRON; }

private:
  EBEECutValues slopeTerm_;
  EBEECutValues slopeStart_;
  EBEECutValues constTerm_;
  bool useHEEPIso_;

  edm::Handle<double> rhoHandle_;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleTrkPtIsoCut, "GsfEleTrkPtIsoCut");

GsfEleTrkPtIsoCut::GsfEleTrkPtIsoCut(const edm::ParameterSet& params)
    : CutApplicatorBase(params),
      slopeTerm_(params, "slopeTerm"),
      slopeStart_(params, "slopeStart"),
      constTerm_(params, "constTerm"),
      useHEEPIso_(params.getParameter<bool>("useHEEPIso")) {}

CutApplicatorBase::result_type GsfEleTrkPtIsoCut::operator()(const reco::GsfElectronPtr& cand) const {
  const float isolTrkPt = useHEEPIso_ ? cand->dr03TkSumPtHEEP() : cand->dr03TkSumPt();

  const float et = cand->et();
  const float cutValue =
      et > slopeStart_(cand) ? slopeTerm_(cand) * (et - slopeStart_(cand)) + constTerm_(cand) : constTerm_(cand);
  return isolTrkPt < cutValue;
}

double GsfEleTrkPtIsoCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  return useHEEPIso_ ? ele->dr03TkSumPtHEEP() : ele->dr03TkSumPt();
}
