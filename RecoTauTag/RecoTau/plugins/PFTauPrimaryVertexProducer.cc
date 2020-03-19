#include "RecoTauTag/RecoTau/interface/PFTauPrimaryVertexProducerBase.h"
#include "FWCore/Framework/interface/MakerMacros.h"

/// RECO/AOD implementation of the PFTauPrimaryVertexProducer plugin
class PFTauPrimaryVertexProducer final : public PFTauPrimaryVertexProducerBase {
public:
  explicit PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig);
  ~PFTauPrimaryVertexProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void nonTauTracksInPV(const reco::VertexRef&,
                        const std::vector<edm::Ptr<reco::TrackBase> >&,
                        std::vector<const reco::Track*>&) override;
};

PFTauPrimaryVertexProducer::PFTauPrimaryVertexProducer(const edm::ParameterSet& iConfig)
    : PFTauPrimaryVertexProducerBase::PFTauPrimaryVertexProducerBase(iConfig) {}

PFTauPrimaryVertexProducer::~PFTauPrimaryVertexProducer() {}

void PFTauPrimaryVertexProducer::nonTauTracksInPV(const reco::VertexRef& thePVRef,
                                                  const std::vector<edm::Ptr<reco::TrackBase> >& tauTracks,
                                                  std::vector<const reco::Track*>& nonTauTracks) {
  //Find non-tau tracks associated to thePV
  for (reco::Vertex::trackRef_iterator vtxTrkRef = thePVRef->tracks_begin(); vtxTrkRef != thePVRef->tracks_end();
       vtxTrkRef++) {
    bool matched = false;
    for (const auto& tauTrack : tauTracks) {
      if (tauTrack.id() == vtxTrkRef->id() && tauTrack.key() == vtxTrkRef->key()) {
        matched = true;
        break;
      }
    }
    if (!matched)
      nonTauTracks.push_back((*vtxTrkRef).get());
  }
}

void PFTauPrimaryVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  auto desc = PFTauPrimaryVertexProducerBase::getDescriptionsBase();
  descriptions.add("pfTauPrimaryVertexProducer", desc);
}

DEFINE_FWK_MODULE(PFTauPrimaryVertexProducer);
