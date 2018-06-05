#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"


class GsfEleRelPFIsoScaledCut : public CutApplicatorWithEventContentBase {
  public:
    GsfEleRelPFIsoScaledCut(const edm::ParameterSet& c);
    
    result_type operator()(const reco::GsfElectronPtr&) const final;

    void setConsumes(edm::ConsumesCollector&) final;
    void getEventContent(const edm::EventBase&) final;

    double value(const reco::CandidatePtr& cand) const final;

    CandidateType candidateType() const final { 
      return ELECTRON; 
    }

  private:
    const float _barrelC0, _endcapC0, _barrelCpt, _endcapCpt, _barrelCutOff;
    bool _isRelativeIso;
    EffectiveAreas _effectiveAreas;
    edm::Handle<double> _rhoHandle;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory, GsfEleRelPFIsoScaledCut, "GsfEleRelPFIsoScaledCut");

GsfEleRelPFIsoScaledCut::GsfEleRelPFIsoScaledCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _barrelC0(c.getParameter<double>("barrelC0")),
  _endcapC0(c.getParameter<double>("endcapC0")),
  _barrelCpt(c.getParameter<double>("barrelCpt")),
  _endcapCpt(c.getParameter<double>("endcapCpt")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _effectiveAreas( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())
{
  edm::InputTag rhoTag = c.getParameter<edm::InputTag>("rho");    
  contentTags_.emplace("rho",rhoTag);  
}

void GsfEleRelPFIsoScaledCut::setConsumes(edm::ConsumesCollector& cc){
  auto rho = cc.consumes<double>(contentTags_["rho"]);
  contentTokens_.emplace("rho", rho);
}

void GsfEleRelPFIsoScaledCut::getEventContent(const edm::EventBase& ev){
  ev.getByLabel(contentTags_["rho"],_rhoHandle);
}

CutApplicatorBase::result_type GsfEleRelPFIsoScaledCut::operator()(const reco::GsfElectronPtr& cand) const {  
  // Establish the cut value
  double absEta = std::abs(cand->superCluster()->eta());

  const float C0     = (absEta < _barrelCutOff ? _barrelC0  : _endcapC0);
  const float Cpt    = (absEta < _barrelCutOff ? _barrelCpt : _endcapCpt);
  const float isoCut = C0+Cpt/cand->pt();

  return value(cand) < isoCut;
}

double GsfEleRelPFIsoScaledCut::value(const reco::CandidatePtr& cand) const {

  // Establish the cut value
  reco::GsfElectronPtr ele(cand);
  double absEta = std::abs(ele->superCluster()->eta());
  
  // Compute the combined isolation with effective area correction
  auto pfIso = ele->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho  = pfIso.sumPhotonEt;
  const float  eA  = _effectiveAreas.getEffectiveArea(absEta);
  const float rho  = _rhoHandle.isValid() ? (float)(*_rhoHandle) : 0; // std::max likes float arguments
  const float iso  = chad + std::max(0.0f, nhad + pho - rho*eA);
  return iso/cand->pt();
}
