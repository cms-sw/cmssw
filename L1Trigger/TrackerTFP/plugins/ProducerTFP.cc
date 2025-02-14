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
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"
#include "L1Trigger/TrackerTFP/interface/TrackFindingProcessor.h"

#include <string>
#include <numeric>
#include <vector>
#include <deque>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerTFP
   *  \brief  L1TrackTrigger final TFP output formatter
   *  \author Thomas Schuh
   *  \date   2023, June
   */
  class ProducerTFP : public stream::EDProducer<> {
  public:
    explicit ProducerTFP(const ParameterSet&);
    ~ProducerTFP() override {}

  private:
    void produce(Event&, const EventSetup&) override;
    // ED input token of stubs and tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<TTTracks> edPutTokenTTTracks_;
    EDPutTokenT<StreamsTrack> edPutTokenTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // TrackQuality token
    ESGetToken<TrackQuality, DataFormatsRcd> esGetTokenTrackQuality_;
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
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(labelStubs, branchStubs));
    edPutTokenTTTracks_ = produces<TTTracks>(branchTTTracks);
    edPutTokenTracks_ = produces<StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenTrackQuality_ = esConsumes<TrackQuality, DataFormatsRcd, Transition::BeginRun>();
  }

  void ProducerTFP::produce(Event& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to determine track quality
    const TrackQuality* trackQuality = &iSetup.getData(esGetTokenTrackQuality_);
    // empty TFP products
    TTTracks ttTracks;
    StreamsTrack streamsTrack(setup->numRegions() * setup->tfpNumChannel());
    // read in TQ Products
    const StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    const StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    // produce TTTracks
    TrackFindingProcessor tfp(setup, dataFormats, trackQuality);
    tfp.produce(tracks, stubs, ttTracks, streamsTrack);
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

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerTFP);