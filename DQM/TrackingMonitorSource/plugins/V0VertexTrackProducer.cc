#include <memory>
#include <vector>
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

class V0VertexTrackProducer : public edm::global::EDProducer<> {
public:
  explicit V0VertexTrackProducer(const edm::ParameterSet&);
  ~V0VertexTrackProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> vccToken_;
};

V0VertexTrackProducer::V0VertexTrackProducer(const edm::ParameterSet& iConfig)
    : vccToken_{consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getParameter<edm::InputTag>("vertexCompositeCandidates"))} {
  produces<std::vector<reco::Track>>();
}

void V0VertexTrackProducer::produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  edm::Handle<reco::VertexCompositeCandidateCollection> vccHandle;
  iEvent.getByToken(vccToken_, vccHandle);

  std::unique_ptr<std::vector<reco::Track>> outputTracks(new std::vector<reco::Track>());

  if (vccHandle.isValid()) {
    // Loop over VertexCompositeCandidates and associate tracks
    for (const auto& vcc : *vccHandle) {
      for (size_t i = 0; i < vcc.numberOfDaughters(); ++i) {
        const reco::Candidate* daughter = vcc.daughter(i);
        const reco::RecoChargedCandidate* chargedDaughter = dynamic_cast<const reco::RecoChargedCandidate*>(daughter);
        if (chargedDaughter) {
          const reco::TrackRef trackRef = chargedDaughter->track();
          if (trackRef.isNonnull()) {
            outputTracks->push_back(*trackRef);
          }
        }
      }
    }
  } else {
    edm::LogError("V0VertexTrackProducer") << "Error >> Failed to get VertexCompositeCandidateCollection";
  }
  iEvent.put(std::move(outputTracks));
}

void V0VertexTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCompositeCandidates", edm::InputTag("generalV0Candidates:Kshort"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(V0VertexTrackProducer);
