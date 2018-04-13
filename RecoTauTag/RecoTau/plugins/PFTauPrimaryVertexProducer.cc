#include "RecoTauTag/RecoTau/interface/PFTauPrimaryVertexProducerBase.h"

/// RECO/AOD implementation of the PFTauPrimaryVertexProducer plugin
class PFTauPrimaryVertexProducer final : public PFTauPrimaryVertexProducerBase {

 public:
  explicit PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFTauPrimaryVertexProducer() override;

 protected:
  void nonTauTracksInPV(const reco::VertexRef&,
			const std::vector<edm::Ptr<reco::TrackBase> >&,
			std::vector<const reco::Track*>&) override;

};

PFTauPrimaryVertexProducer::PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig): 
  PFTauPrimaryVertexProducerBase::PFTauPrimaryVertexProducerBase(iConfig) {}

PFTauPrimaryVertexProducer::~PFTauPrimaryVertexProducer(){}

void PFTauPrimaryVertexProducer::nonTauTracksInPV(const reco::VertexRef &thePVRef,
						  const std::vector<edm::Ptr<reco::TrackBase> > &tauTracks,
						  std::vector<const reco::Track*> &nonTauTracks){

  //Find non-tau tracks associated to thePV
  for(reco::Vertex::trackRef_iterator vtxTrkRef=thePVRef->tracks_begin();vtxTrkRef!=thePVRef->tracks_end();vtxTrkRef++){
    bool matched = false;
    edm::Ptr<reco::TrackBase> vtxTrkPtr = edm::refToPtr((*vtxTrkRef).castTo<edm::Ref<std::vector<reco::TrackBase> > >());
    for(const auto& tauTrack: tauTracks){
      if( vtxTrkPtr == tauTrack ) {
	matched = true;
	break;
      }
    }
    if( !matched ) nonTauTracks.push_back((*vtxTrkRef).get());
  }
}

DEFINE_FWK_MODULE(PFTauPrimaryVertexProducer);
