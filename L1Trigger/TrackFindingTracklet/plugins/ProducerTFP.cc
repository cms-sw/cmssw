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

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trklet {

  /*! \class  trklet::ProducerTFP
   *  \brief  L1TrackTrigger final TFP output formatter
   *  \author Thomas Schuh
   *  \date   2023, June
   */
  class ProducerTFP : public stream::EDProducer<> {
  public:
    explicit ProducerTFP(const ParameterSet&);
    ~ProducerTFP() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    // ED input token of stubs and tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    EDGetTokenT<Streams> edGetTokenTracksAdd_;
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<TTTracks> edPutTokenTTTracks_;
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
    // helper class to determine track quality
    const TrackQuality* trackQuality_ = nullptr;
  };

  ProducerTFP::ProducerTFP(const ParameterSet& iConfig) {
    const string& labelTracks = iConfig.getParameter<string>("InputLabelTFP");
    const string& labelStubs = iConfig.getParameter<string>("InputLabelTQ");
    const string& branchTracks = iConfig.getParameter<string>("BranchTracks");
    const string& branchTTTracks = iConfig.getParameter<string>("BranchTTTracks");
    const string& branchStubs = iConfig.getParameter<string>("BranchStubs");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(labelTracks, branchTracks));
    edGetTokenTracksAdd_ = consumes<Streams>(InputTag(labelTracks, branchTracks));
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(labelStubs, branchStubs));
    edPutTokenTTTracks_ = produces<TTTracks>(branchTTTracks);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenTrackQuality_ = esConsumes<TrackQuality, TrackQualityRcd, Transition::BeginRun>();
  }

  void ProducerTFP::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to determine track quality
    trackQuality_ = &iSetup.getData(esGetTokenTrackQuality_);
  }

  void ProducerTFP::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty TFP products
    TTTracks ttTracks;
    StreamsTrack streamsTrack(setup_->numRegions() * setup_->tfpNumChannel());
    // read in TQ Products
    const StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    const Streams& tracksAdd = iEvent.get(edGetTokenTracksAdd_);
    const StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    // produce TTTracks
    TrackFindingProcessor tfp(setup_, dataFormats_, trackQuality_);
    tfp.produce(tracks, tracksAdd, stubs, ttTracks, streamsTrack);
    // put TTTRacks and produce TTTRackRefs
    const int nTrks = ttTracks.size();
    const OrphanHandle<TTTracks> oh = iEvent.emplace(edPutTokenTTTracks_, std::move(ttTracks));
    vector<TTTrackRef> ttTrackRefs;
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