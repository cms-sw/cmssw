#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackFindingProcessor.h"

#include <string>
#include <numeric>
#include <vector>
#include <deque>

namespace trklet {

  /*! \class  trklet::ProducerTFP
   *  \brief  L1TrackTrigger final TFP output formatter
   *  \author Thomas Schuh
   *  \date   2023, June
   */
  class ProducerTFP : public edm::stream::EDProducer<> {
  public:
    explicit ProducerTFP(const edm::ParameterSet&);
    ~ProducerTFP() override {}

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of stubs and tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    edm::EDGetTokenT<tt::Streams> edGetTokenTracksAdd_;
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED output token for accepted stubs and tracks
    edm::EDPutTokenT<tt::TTTracks> edPutTokenTTTracks_;
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, ChannelAssignmentRcd> esGetTokenDataFormats_;
    // TrackQuality token
    edm::ESGetToken<trackerTFP::TrackQuality, trackerTFP::DataFormatsRcd> esGetTokenTrackQuality_;
  };

  ProducerTFP::ProducerTFP(const edm::ParameterSet& iConfig) {
    const std::string& labelTracks = iConfig.getParameter<std::string>("InputLabelTFP");
    const std::string& labelStubs = iConfig.getParameter<std::string>("InputLabelTQ");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    const std::string& branchTTTracks = iConfig.getParameter<std::string>("BranchTTTracks");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<tt::StreamsTrack>(edm::InputTag(labelTracks, branchTracks));
    edGetTokenTracksAdd_ = consumes<tt::Streams>(edm::InputTag(labelTracks, branchTracks));
    edGetTokenStubs_ = consumes<tt::StreamsStub>(edm::InputTag(labelStubs, branchStubs));
    edPutTokenTTTracks_ = produces<tt::TTTracks>(branchTTTracks);
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenTrackQuality_ = esConsumes();
  }

  void ProducerTFP::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to determine track quality
    const trackerTFP::TrackQuality* trackQuality = &iSetup.getData(esGetTokenTrackQuality_);
    // empty TFP products
    tt::TTTracks ttTracks;
    tt::StreamsTrack streamsTrack(setup->numRegions() * setup->tfpNumChannel());
    // read in TQ Products
    const tt::StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    const tt::Streams& tracksAdd = iEvent.get(edGetTokenTracksAdd_);
    const tt::StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    // produce TTTracks
    TrackFindingProcessor tfp(setup, dataFormats, trackQuality);
    tfp.produce(tracks, tracksAdd, stubs, ttTracks, streamsTrack);
    // put TTTRacks and produce TTTRackRefs
    const int nTrks = ttTracks.size();
    const edm::OrphanHandle<tt::TTTracks> oh = iEvent.emplace(edPutTokenTTTracks_, std::move(ttTracks));
    std::vector<TTTrackRef> ttTrackRefs;
    ttTrackRefs.reserve(nTrks);
    for (int iTrk = 0; iTrk < nTrks; iTrk++)
      ttTrackRefs.emplace_back(oh, iTrk);
    // replace old TTTrackRefs in streamsTrack with new TTTrackRefs
    tfp.produce(ttTrackRefs, streamsTrack);
    // put StreamsTrack
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerTFP);
