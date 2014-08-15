#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleDzCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleDzCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:  
  const double _dzCutValueEB,_dzCutValueEE,_barrelCutOff;
  edm::Handle<reco::VertexCollection> _vtxs;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDzCut,
		  "GsfEleDzCut");

GsfEleDzCut::GsfEleDzCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _dzCutValueEB(c.getParameter<double>("dzCutValueEB")),
  _dzCutValueEE(c.getParameter<double>("dzCutValueEE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  edm::InputTag vertextag = c.getParameter<edm::InputTag>("vertexSrc");
  contentTags_.emplace("vertices",vertextag);
}

void GsfEleDzCut::setConsumes(edm::ConsumesCollector& cc) {
  auto vtcs = 
    cc.consumes<reco::VertexCollection>(contentTags_["vertices"]);
  contentTokens_.emplace("vertices",vtcs);
}

void GsfEleDzCut::getEventContent(const edm::EventBase& ev) {    
  ev.getByLabel(contentTags_["vertices"],_vtxs);
}

CutApplicatorBase::result_type 
GsfEleDzCut::
operator()(const reco::GsfElectronRef& cand) const{  
  const float dzCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _dzCutValueEB : _dzCutValueEE );
  
  const reco::VertexCollection& vtxs = *_vtxs;
  const double dz = ( vtxs.size() ? 
		      cand->gsfTrack()->dz(vtxs[0].position()) : 
		      cand->gsfTrack()->dz() );
  return std::abs(dz) < dzCutValue;
}
