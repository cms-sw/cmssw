#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class PhoAnyPFIsoWithEACut : public CutApplicatorWithEventContentBase {
public:
  PhoAnyPFIsoWithEACut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::PhotonPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return PHOTON; 
  }

private:
  float _anyPFIsoWithEACutValue_C1_EB;
  float _anyPFIsoWithEACutValue_C2_EB;
  float _anyPFIsoWithEACutValue_C1_EE;
  float _anyPFIsoWithEACutValue_C2_EE;
  float _barrelCutOff;
  bool  _useRelativeIso;
  edm::Handle<edm::ValueMap<float> > _anyPFIsoWithEAMap;

  constexpr static char anyPFIsoWithEA_[] = "anyPFIsoWithEA";
};

constexpr char PhoAnyPFIsoWithEACut::anyPFIsoWithEA_[];

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoAnyPFIsoWithEACut,
		  "PhoAnyPFIsoWithEACut");

PhoAnyPFIsoWithEACut::PhoAnyPFIsoWithEACut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _anyPFIsoWithEACutValue_C1_EB(c.getParameter<double>("anyPFIsoWithEACutValue_C1_EB")),
  _anyPFIsoWithEACutValue_C2_EB(c.getParameter<double>("anyPFIsoWithEACutValue_C2_EB")),
  _anyPFIsoWithEACutValue_C1_EE(c.getParameter<double>("anyPFIsoWithEACutValue_C1_EE")),
  _anyPFIsoWithEACutValue_C2_EE(c.getParameter<double>("anyPFIsoWithEACutValue_C2_EE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")),
  _useRelativeIso(c.getParameter<bool>("useRelativeIso")) {
  
  edm::InputTag maptag = c.getParameter<edm::InputTag>("anyPFIsoWithEAMap");
  contentTags_.emplace(anyPFIsoWithEA_,maptag);
}

void PhoAnyPFIsoWithEACut::setConsumes(edm::ConsumesCollector& cc) {
  auto anyPFIsoWithEA = 
    cc.consumes<edm::ValueMap<float> >(contentTags_[anyPFIsoWithEA_]);
  contentTokens_.emplace(anyPFIsoWithEA_,anyPFIsoWithEA);
}

void PhoAnyPFIsoWithEACut::getEventContent(const edm::EventBase& ev) {  
  ev.getByLabel(contentTags_[anyPFIsoWithEA_],_anyPFIsoWithEAMap);
}

CutApplicatorBase::result_type 
PhoAnyPFIsoWithEACut::
operator()(const reco::PhotonPtr& cand) const{  

  // Figure out the cut value
  // The value is generally pt-dependent: C1 + pt * C2
  const float anyPFIsoWithEACutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _anyPFIsoWithEACutValue_C1_EB + cand->pt() * _anyPFIsoWithEACutValue_C2_EB
      : 
      _anyPFIsoWithEACutValue_C1_EE + cand->pt() * _anyPFIsoWithEACutValue_C2_EE
      );
  
  // Retrieve the variable value for this particle
  float anyPFIsoWithEA = (*_anyPFIsoWithEAMap)[cand];

  // Divide by pT if the relative isolation is requested
  if( _useRelativeIso )
    anyPFIsoWithEA /= cand->pt();

  // Apply the cut and return the result
  return anyPFIsoWithEA < anyPFIsoWithEACutValue;
}
