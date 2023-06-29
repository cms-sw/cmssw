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

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerKFin
   *  \brief  transforms TTTracks into KF input
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerKFin : public stream::EDProducer<> {
  public:
    explicit ProducerKFin(const ParameterSet&);
    ~ProducerKFin() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of TTTracks
    EDGetTokenT<vector<TTTrack<Ref_Phase2TrackerDigi_>>> edGetTokenTTTracks_;
    // ED input token of Stubs
    EDGetTokenT<StreamsStub> edGetTokenStubs_;
    // ED output token for stubs
    EDPutTokenT<StreamsStub> edPutTokenAcceptedStubs_;
    EDPutTokenT<StreamsStub> edPutTokenLostStubs_;
    // ED output token for tracks
    EDPutTokenT<StreamsTrack> edPutTokenAcceptedTracks_;
    EDPutTokenT<StreamsTrack> edPutTokenLostTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // helper class to encode layer
    const LayerEncoding* layerEncoding_ = nullptr;
    //
    bool enableTruncation_;
  };

  ProducerKFin::ProducerKFin(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& labelTTTracks = iConfig.getParameter<string>("LabelZHTout");
    const string& labelStubs = iConfig.getParameter<string>("LabelZHT");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLostStubs = iConfig.getParameter<string>("BranchLostStubs");
    const string& branchLostTracks = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenTTTracks_ =
        consumes<vector<TTTrack<Ref_Phase2TrackerDigi_>>>(InputTag(labelTTTracks, branchAcceptedTracks));
    edGetTokenStubs_ = consumes<StreamsStub>(InputTag(labelStubs, branchAcceptedStubs));
    edPutTokenAcceptedStubs_ = produces<StreamsStub>(branchAcceptedStubs);
    edPutTokenAcceptedTracks_ = produces<StreamsTrack>(branchAcceptedTracks);
    edPutTokenLostStubs_ = produces<StreamsStub>(branchLostStubs);
    edPutTokenLostTracks_ = produces<StreamsTrack>(branchLostTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
    //
    enableTruncation_ = iConfig.getParameter<bool>("EnableTruncation");
  }

  void ProducerKFin::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
  }

  void ProducerKFin::produce(Event& iEvent, const EventSetup& iSetup) {
    const DataFormat& dfcot = dataFormats_->format(Variable::cot, Process::kfin);
    const DataFormat& dfzT = dataFormats_->format(Variable::zT, Process::kfin);
    const DataFormat& dfinv2R = dataFormats_->format(Variable::inv2R, Process::kfin);
    const DataFormat& dfdPhi = dataFormats_->format(Variable::dPhi, Process::kfin);
    const DataFormat& dfdZ = dataFormats_->format(Variable::dZ, Process::kfin);
    // empty KFin products
    StreamsStub streamAcceptedStubs(dataFormats_->numStreamsStubs(Process::kf));
    StreamsTrack streamAcceptedTracks(dataFormats_->numStreamsTracks(Process::kf));
    StreamsStub streamLostStubs(dataFormats_->numStreamsStubs(Process::kf));
    StreamsTrack streamLostTracks(dataFormats_->numStreamsTracks(Process::kf));
    // read in SFout Product and produce KFin product
    if (setup_->configurationSupported()) {
      Handle<StreamsStub> handleStubs;
      iEvent.getByToken<StreamsStub>(edGetTokenStubs_, handleStubs);
      const StreamsStub& streams = *handleStubs.product();
      Handle<vector<TTTrack<Ref_Phase2TrackerDigi_>>> handleTTTracks;
      iEvent.getByToken<vector<TTTrack<Ref_Phase2TrackerDigi_>>>(edGetTokenTTTracks_, handleTTTracks);
      const vector<TTTrack<Ref_Phase2TrackerDigi_>>& ttTracks = *handleTTTracks.product();
      for (int region = 0; region < setup_->numRegions(); region++) {
        // Unpack input SF data into vector
        int nStubsZHR(0);
        for (int channel = 0; channel < dataFormats_->numChannel(Process::zht); channel++) {
          const int index = region * dataFormats_->numChannel(Process::zht) + channel;
          const StreamStub& stream = streams[index];
          nStubsZHR += accumulate(stream.begin(), stream.end(), 0, [](int sum, const FrameStub& frame) {
            return sum + ( frame.first.isNonnull() ? 1 : 0 );
          });
        }
        vector<StubZHT> stubsZHT;
        stubsZHT.reserve(nStubsZHR);
        for (int channel = 0; channel < dataFormats_->numChannel(Process::zht); channel++) {
          const int index = region * dataFormats_->numChannel(Process::zht) + channel;
          for (const FrameStub& frame : streams[index])
            if (frame.first.isNonnull())
              stubsZHT.emplace_back(frame, dataFormats_);
        }
        vector<deque<FrameStub>> dequesStubs(dataFormats_->numChannel(Process::kf) * setup_->numLayers());
        vector<deque<FrameTrack>> dequesTracks(dataFormats_->numChannel(Process::kf));
        int i(0);
        for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks) {
          if ((int)ttTrack.phiSector() / setup_->numSectorsPhi() != region) {
            i++;
            continue;
          }
          const int sectorPhi = ttTrack.phiSector() % setup_->numSectorsPhi();
          deque<FrameTrack>& tracks = dequesTracks[sectorPhi];
          const int binEta = ttTrack.etaSector();
          const int binZT = dfzT.toUnsigned(dfzT.integer(ttTrack.z0()));
          const int binCot = dfcot.toUnsigned(dfcot.integer(ttTrack.tanL()));
          StubZHT* stubZHT = nullptr;
          vector<int> layerCounts(setup_->numLayers(), 0);
          for (const TTStubRef& ttStubRef : ttTrack.getStubRefs()) {
            const int layerId = setup_->layerId(ttStubRef);
            const int layerIdKF = layerEncoding_->layerIdKF(binEta, binZT, binCot, layerId);
            if (layerIdKF == -1)
              continue;
            if (layerCounts[layerIdKF] == setup_->zhtMaxStubsPerLayer())
              continue;
            layerCounts[layerIdKF]++;
            deque<FrameStub>& stubs = dequesStubs[sectorPhi * setup_->numLayers() + layerIdKF];
            auto identical = [ttStubRef, ttTrack](const StubZHT& stub) {
              return (int)ttTrack.trackSeedType() == stub.trackId() && ttStubRef == stub.ttStubRef();
            };
            stubZHT = &*find_if(stubsZHT.begin(), stubsZHT.end(), identical);
            const double inv2R = dfinv2R.floating(stubZHT->inv2R());
            const double cot = dfcot.floating(stubZHT->cot()) + setup_->sectorCot(binEta);
            const double dPhi = dfdPhi.digi(setup_->dPhi(ttStubRef, inv2R));
            const double dZ = dfdZ.digi(setup_->dZ(ttStubRef, cot));
            stubs.emplace_back(StubKFin(*stubZHT, dPhi, dZ, layerIdKF).frame());
          }
          const int size = *max_element(layerCounts.begin(), layerCounts.end());
          int layerIdKF(0);
          for (int layerCount : layerCounts) {
            deque<FrameStub>& stubs = dequesStubs[sectorPhi * setup_->numLayers() + layerIdKF++];
            const int nGaps = size - layerCount;
            stubs.insert(stubs.end(), nGaps, FrameStub());
          }
          const TTBV& maybePattern = layerEncoding_->maybePattern(binEta, binZT, binCot);
          const TrackKFin track(*stubZHT, TTTrackRef(handleTTTracks, i++), maybePattern);
          tracks.emplace_back(track.frame());
          const int nGaps = size - 1;
          tracks.insert(tracks.end(), nGaps, FrameTrack());
        }
        // transform deques to vectors & emulate truncation
        for (int channel = 0; channel < dataFormats_->numChannel(Process::kf); channel++) {
          const int index = region * dataFormats_->numChannel(Process::kf) + channel;
          deque<FrameTrack>& tracks = dequesTracks[channel];
          auto limitTracks = next(tracks.begin(), min(setup_->numFrames(), (int)tracks.size()));
          if (!enableTruncation_)
            limitTracks = tracks.end();
          streamAcceptedTracks[index] = StreamTrack(tracks.begin(), limitTracks);
          streamLostTracks[index] = StreamTrack(limitTracks, tracks.end());
          for (int l = 0; l < setup_->numLayers(); l++) {
            deque<FrameStub>& stubs = dequesStubs[channel * setup_->numLayers() + l];
            auto limitStubs = next(stubs.begin(), min(setup_->numFrames(), (int)stubs.size()));
            if (!enableTruncation_)
              limitStubs = stubs.end();
            streamAcceptedStubs[index * setup_->numLayers() + l] = StreamStub(stubs.begin(), limitStubs);
            streamLostStubs[index * setup_->numLayers() + l] = StreamStub(limitStubs, stubs.end());
          }
        }
      }
    }
    // store products
    iEvent.emplace(edPutTokenAcceptedStubs_, std::move(streamAcceptedStubs));
    iEvent.emplace(edPutTokenAcceptedTracks_, std::move(streamAcceptedTracks));
    iEvent.emplace(edPutTokenLostStubs_, std::move(streamLostStubs));
    iEvent.emplace(edPutTokenLostTracks_, std::move(streamLostTracks));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerKFin);
