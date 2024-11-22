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

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerTQ
   *  \brief  Bit accurate emulation of the track quality BDT
   *  \author Thomas Schuh
   *  \date   2024, Aug
   */
  class ProducerTQ : public stream::EDProducer<> {
  public:
    explicit ProducerTQ(const ParameterSet&);
    ~ProducerTQ() override {}
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}

  private:
    typedef TrackQuality::Track Track;
    // ED input token of kf stubs
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED input token of kf tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted kfout tracks
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // TrackQuality token
    ESGetToken<TrackQuality, TrackQualityRcd> esGetTokenTrackQuality_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // helper class to determine Track Quality
    const TrackQuality* trackQuality_ = nullptr;
  };

  ProducerTQ::ProducerTQ(const ParameterSet& iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelTQ");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenTrackQuality_ = esConsumes<TrackQuality, TrackQualityRcd, Transition::BeginRun>();
  }

  void ProducerTQ::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to determine Track Quality
    trackQuality_ = &iSetup.getData(esGetTokenTrackQuality_);
  }

  void ProducerTQ::produce(Event& iEvent, const EventSetup& iSetup) {
    static const int numRegions = setup_->numRegions();
    static const int numLayers = setup_->numLayers();
    static const int numChannel = setup_->tqNumChannel();
    auto valid = [](int sum, const FrameTrack& frame) { return sum += (frame.first.isNull() ? 0 : 1); };
    // empty TQ product
    StreamsTrack output(numRegions * numChannel);
    // read in KF Product and produce TQ product
    Handle<StreamsStub> handleStubs;
    iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
    const StreamsStub& streamsStubs = *handleStubs.product();
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
    const StreamsTrack& streamsTracks = *handleTracks.product();
    for (int region = 0; region < numRegions; region++) {
      // calculate track quality
      const int offsetLayer = region * numLayers;
      const StreamTrack& streamTrack = streamsTracks[region];
      const int nTracks = accumulate(streamTrack.begin(), streamTrack.end(), 0, valid);
      vector<Track> tracks;
      tracks.reserve(nTracks);
      vector<Track*> stream;
      stream.reserve(streamTrack.size());
      for (int frame = 0; frame < (int)streamTrack.size(); frame++) {
        const FrameTrack& frameTrack = streamTrack[frame];
        if (frameTrack.first.isNull()) {
          stream.push_back(nullptr);
          continue;
        }
        StreamStub streamStub;
        streamStub.reserve(numLayers);
        for (int layer = 0; layer < numLayers; layer++)
          streamStub.push_back(streamsStubs[offsetLayer + layer][frame]);
        tracks.emplace_back(frameTrack, streamStub, trackQuality_);
        stream.push_back(&tracks.back());
      }
      // fill TQ product
      const int offsetChannel = region * numChannel;
      for (int channel = 0; channel < numChannel; channel++)
        output[offsetChannel + channel].reserve(stream.size());
      for (Track* track : stream) {
        if (!track) {
          for (int channel = 0; channel < numChannel; channel++)
            output[offsetChannel + channel].emplace_back(FrameTrack());
          continue;
        }
        for (int channel = 0; channel < numChannel; channel++)
          output[offsetChannel + channel].emplace_back(track->frame(channel));
      }
    }
    // store TQ product
    iEvent.emplace(edPutTokenTracks_, std::move(output));
  }
}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerTQ);
