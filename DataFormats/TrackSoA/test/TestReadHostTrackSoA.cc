#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/TrackSoA/interface/TracksHost.h"

namespace edmtest {

  class TestReadHostTrackSoA : public edm::global::EDAnalyzer<> {
  public:
    TestReadHostTrackSoA(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::EDGetTokenT<::reco::TracksHost> getToken_;
    const unsigned int trackSize_;
  };

  TestReadHostTrackSoA::TestReadHostTrackSoA(edm::ParameterSet const& iPSet)
      : getToken_(consumes(iPSet.getParameter<edm::InputTag>("input"))),
        trackSize_(iPSet.getParameter<unsigned int>("trackSize")) {}

  void TestReadHostTrackSoA::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& tracks = iEvent.get(getToken_);
    auto tracksView = tracks.view();

    assert(tracksView.metadata().size() == int(trackSize_));
    assert(tracksView.nTracks() == int(trackSize_));

    for (int i = 0; i < tracksView.metadata().size(); ++i) {
      if (tracksView[i].eta() != float(i)) {
        throw cms::Exception("TestReadHostTrackSoA Failure") << "TestReadHostTrackSoA::analyze, entry. i = " << i;
      }
    }
  }

  void TestReadHostTrackSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("input");
    desc.add<unsigned int>("trackSize", 1000);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadHostTrackSoA;
DEFINE_FWK_MODULE(TestReadHostTrackSoA);
