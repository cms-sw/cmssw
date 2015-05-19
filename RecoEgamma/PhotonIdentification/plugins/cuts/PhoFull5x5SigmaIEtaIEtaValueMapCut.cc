#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhoFull5x5SigmaIEtaIEtaValueMapCut : public CutApplicatorWithEventContentBase {
public:
  PhoFull5x5SigmaIEtaIEtaValueMapCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::PhotonPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return PHOTON; 
  }

private:
  float _cutValueEB;
  float _cutValueEE;
  float _barrelCutOff;
  edm::Handle<edm::ValueMap<float> > _full5x5SigmaIEtaIEtaMap;

  constexpr static char full5x5SigmaIEtaIEta_[] = "full5x5SigmaIEtaIEta";
};

constexpr char PhoFull5x5SigmaIEtaIEtaValueMapCut::full5x5SigmaIEtaIEta_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoFull5x5SigmaIEtaIEtaValueMapCut,
		  "PhoFull5x5SigmaIEtaIEtaValueMapCut");

PhoFull5x5SigmaIEtaIEtaValueMapCut::PhoFull5x5SigmaIEtaIEtaValueMapCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _cutValueEB(c.getParameter<double>("cutValueEB")),
  _cutValueEE(c.getParameter<double>("cutValueEE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")) {

  edm::InputTag maptag = c.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap");
  contentTags_.emplace(full5x5SigmaIEtaIEta_,maptag);
}

void PhoFull5x5SigmaIEtaIEtaValueMapCut::setConsumes(edm::ConsumesCollector& cc) {
  auto full5x5SigmaIEtaIEta = 
    cc.consumes<edm::ValueMap<float> >(contentTags_[full5x5SigmaIEtaIEta_]);
  contentTokens_.emplace(full5x5SigmaIEtaIEta_,full5x5SigmaIEtaIEta);
}

void PhoFull5x5SigmaIEtaIEtaValueMapCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_[full5x5SigmaIEtaIEta_],_full5x5SigmaIEtaIEtaMap);
}

CutApplicatorBase::result_type 
PhoFull5x5SigmaIEtaIEtaValueMapCut::
operator()(const reco::PhotonPtr& cand) const{  

  // Figure out the cut value
  const float cutValue = 
    ( std::abs(cand->superCluster()->eta()) < _barrelCutOff ? 
      _cutValueEB : _cutValueEE );
  
  // Retrieve the variable value for this particle
  const float full5x5SigmaIEtaIEta = (*_full5x5SigmaIEtaIEtaMap)[cand];
  
  // Apply the cut and return the result
  return full5x5SigmaIEtaIEta < cutValue;
}
