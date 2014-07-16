#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleDxyCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleDxyCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronRef&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:  
  const double _dxyCutValueEB, _dxyCutValueEE,_barrelCutOff;
  edm::Handle<reco::VertexCollection> _vtxs;
};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleDxyCut,
		  "GsfEleDxyCut");

GsfEleDxyCut::GsfEleDxyCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _dxyCutValueEB(c.getParameter<double>("dxyCutValueEB")),
  _dxyCutValueEE(c.getParameter<double>("dxyCutValueEE")),
  _barrelCutOff(c.getParameter<double>("barrelCutOff")) {
  edm::InputTag vertextag = c.getParameter<edm::InputTag>("vertexSrc");
  contentTags_.emplace("vertices",vertextag);
}

void GsfEleDxyCut::setConsumes(edm::ConsumesCollector& cc) {
  auto vtcs = 
    cc.consumes<reco::VertexCollection>(contentTags_["vertices"]);
  contentTokens_.emplace("vertices",vtcs);
}

void GsfEleDxyCut::getEventContent(const edm::EventBase& ev) {    
  ev.getByLabel(contentTags_["vertices"],_vtxs);
}

CutApplicatorBase::result_type 
GsfEleDxyCut::
operator()(const reco::GsfElectronRef& cand) const{  
  const unsigned dxyCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _dxyCutValueEB : _dxyCutValueEE );

  const reco::VertexCollection& vtxs = *_vtxs;
  const double dxy = ( vtxs.size() ? 
		       cand->gsfTrack()->dxy(vtxs[0].position()) : 
		       cand->gsfTrack()->dxy() );
  return dxy < dxyCutValue;
}
