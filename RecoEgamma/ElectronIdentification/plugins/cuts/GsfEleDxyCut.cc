#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

class GsfEleDxyCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleDxyCut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final { 
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
  edm::InputTag vertextagMiniAOD = c.getParameter<edm::InputTag>("vertexSrcMiniAOD"); 
  contentTags_.emplace("vertices",vertextag);
  contentTags_.emplace("verticesMiniAOD",vertextagMiniAOD);
}

void GsfEleDxyCut::setConsumes(edm::ConsumesCollector& cc) {
  auto vtcs = cc.mayConsume<reco::VertexCollection>(contentTags_["vertices"]);
  auto vtcsMiniAOD = cc.mayConsume<reco::VertexCollection>(contentTags_["verticesMiniAOD"]);
  contentTokens_.emplace("vertices",vtcs);
  contentTokens_.emplace("verticesMiniAOD",vtcsMiniAOD);
}

void GsfEleDxyCut::getEventContent(const edm::EventBase& ev) {    
  // First try AOD, then go to miniAOD. Use the same Handle since collection class is the same.
  ev.getByLabel(contentTags_["vertices"],_vtxs);
  if (!_vtxs.isValid())
    ev.getByLabel(contentTags_["verticesMiniAOD"],_vtxs);
}

CutApplicatorBase::result_type 
GsfEleDxyCut::
operator()(const reco::GsfElectronPtr& cand) const{  
  const float dxyCutValue = 
    ( std::abs(cand->superCluster()->position().eta()) < _barrelCutOff ? 
      _dxyCutValueEB : _dxyCutValueEE );

  const reco::VertexCollection& vtxs = *_vtxs;
  const double dxy = ( !vtxs.empty() ? 
		       cand->gsfTrack()->dxy(vtxs[0].position()) : 
		       cand->gsfTrack()->dxy() );
  return std::abs(dxy) < dxyCutValue;
}

double GsfEleDxyCut::value(const reco::CandidatePtr& cand) const {
  reco::GsfElectronPtr ele(cand);
  const reco::VertexCollection& vtxs = *_vtxs;
  const double dxy = ( !vtxs.empty() ? 
		       ele->gsfTrack()->dxy(vtxs[0].position()) : 
		       ele->gsfTrack()->dxy() );
  return std::abs(dxy);
}
