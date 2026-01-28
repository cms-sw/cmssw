#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/TrackSoA/interface/TracksHost.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteHostTrackSoA : public edm::global::EDProducer<> {
  public:
    TestWriteHostTrackSoA(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const unsigned int trackSize_;
    edm::EDPutTokenT<::reco::TracksHost> putToken_;
  };

  TestWriteHostTrackSoA::TestWriteHostTrackSoA(edm::ParameterSet const& iPSet)
      : trackSize_(iPSet.getParameter<unsigned int>("trackSize")), putToken_(produces()) {}

  void TestWriteHostTrackSoA::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    ::reco::TracksHost tracks(cms::alpakatools::host(), trackSize_, 4 * trackSize_);
    auto tracksBlocksView = tracks.view();
    for (unsigned int i = 0; i < trackSize_; ++i) {
      tracksBlocksView.tracks()[i].eta() = float(i);
    }
    tracksBlocksView.tracks().nTracks() = trackSize_;
    iEvent.emplace(putToken_, std::move(tracks));
  }

  void TestWriteHostTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("trackSize", 1000);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteHostTrackSoA;
DEFINE_FWK_MODULE(TestWriteHostTrackSoA);
