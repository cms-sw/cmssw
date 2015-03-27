#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhoFull5x5SigmaIEtaIEtaCut : public CutApplicatorWithEventContentBase {
public:
  PhoFull5x5SigmaIEtaIEtaCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::PhotonPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:
  float _full5x5SigmaIEtaIEtaCutValueEB;
  float _full5x5SigmaIEtaIEtaCutValueEE;
  float _barrelCutOff;
  edm::Handle<edm::ValueMap<float> > _full5x5SigmaIEtaIEtaMap;

  constexpr static char full5x5SigmaIEtaIEta_[] = "full5x5SigmaIEtaIEta";
};

constexpr char PhoFull5x5SigmaIEtaIEtaCut::full5x5SigmaIEtaIEta_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoFull5x5SigmaIEtaIEtaCut,
		  "PhoFull5x5SigmaIEtaIEtaCut");

PhoFull5x5SigmaIEtaIEtaCut::PhoFull5x5SigmaIEtaIEtaCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _full5x5SigmaIEtaIEtaCutValueEB(c.getParameter<double>("full5x5SigmaIEtaIEtaCutValueEB")),
  _full5x5SigmaIEtaIEtaCutValueEE(c.getParameter<double>("full5x5SigmaIEtaIEtaCutValueEE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")) {

  printf("DEBUG: sigmaIetaIeta cut is constructed\n");
  edm::InputTag maptag = c.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap");
  contentTags_.emplace(full5x5SigmaIEtaIEta_,maptag);
}

void PhoFull5x5SigmaIEtaIEtaCut::setConsumes(edm::ConsumesCollector& cc) {
  auto full5x5SigmaIEtaIEta = 
    cc.consumes<edm::ValueMap<float> >(contentTags_[full5x5SigmaIEtaIEta_]);
  contentTokens_.emplace(full5x5SigmaIEtaIEta_,full5x5SigmaIEtaIEta);
}

void PhoFull5x5SigmaIEtaIEtaCut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_[full5x5SigmaIEtaIEta_],_full5x5SigmaIEtaIEtaMap);
}

CutApplicatorBase::result_type 
PhoFull5x5SigmaIEtaIEtaCut::
operator()(const reco::PhotonPtr& cand) const{  

  // Figure out the cut value
  const float full5x5SigmaIEtaIEtaCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _full5x5SigmaIEtaIEtaCutValueEB : _full5x5SigmaIEtaIEtaCutValueEE );
  
  // Retrieve the variable value for this particle
  const float full5x5SigmaIEtaIEta = (*_full5x5SigmaIEtaIEtaMap)[cand];
  printf("DEBUG: see=%f  eta=%f   cut=%f   result= %d\n", 
	 full5x5SigmaIEtaIEta, cand->superCluster()->position().eta(),
	 full5x5SigmaIEtaIEtaCutValue, 
	 (int)(full5x5SigmaIEtaIEta < full5x5SigmaIEtaIEtaCutValue)); fflush(stdout);
  
  // Apply the cut and return the result
  return full5x5SigmaIEtaIEta < full5x5SigmaIEtaIEtaCutValue;
}
