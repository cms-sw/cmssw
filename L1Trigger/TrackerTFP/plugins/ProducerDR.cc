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
#include "L1Trigger/TrackerTFP/interface/DuplicateRemoval.h"

#include <string>
#include <numeric>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerDR
   *  \brief  L1TrackTrigger duplicate removal emulator
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class ProducerDR : public edm::stream::EDProducer<> {
  public:
    explicit ProducerDR(const edm::ParameterSet&);
    ~ProducerDR() override {}

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of sf stubs and tracks
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // number of input channel
    int numChannelIn_;
    // number of output channel
    int numChannelOut_;
    // number ofprocessing regions
    int numRegions_;
    // number of kf layers
    int numLayers_;
  };

  ProducerDR::ProducerDR(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelDR");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<tt::StreamsStub>(edm::InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<tt::StreamsTrack>(edm::InputTag(label, branchTracks));
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
  }

  void ProducerDR::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    numRegions_ = setup->numRegions();
    numLayers_ = setup->numLayers();
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    numChannelIn_ = dataFormats->numChannel(Process::kf);
    numChannelOut_ = dataFormats->numChannel(Process::dr);
  }

  void ProducerDR::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // empty DR products
    tt::StreamsStub acceptedStubs(numRegions_ * numChannelOut_ * numLayers_);
    tt::StreamsTrack acceptedTracks(numRegions_ * numChannelOut_);
    // read in KF Product and produce DR product
    const tt::StreamsStub& allStubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& allTracks = iEvent.get(edGetTokenTracks_);
    // helper
    auto validFrameT = [](int sum, const tt::FrameTrack& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto validFrameS = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto putT = [](const std::vector<TrackDR*>& objects, tt::StreamTrack& stream) {
      auto toFrame = [](TrackDR* object) { return object ? object->frame() : tt::FrameTrack(); };
      stream.reserve(objects.size());
      std::transform(objects.begin(), objects.end(), std::back_inserter(stream), toFrame);
    };
    auto putS = [](const std::vector<StubDR*>& objects, tt::StreamStub& stream) {
      auto toFrame = [](StubDR* object) { return object ? object->frame() : tt::FrameStub(); };
      stream.reserve(objects.size());
      std::transform(objects.begin(), objects.end(), std::back_inserter(stream), toFrame);
    };
    for (int region = 0; region < numRegions_; region++) {
      const int offsetIn = region * numChannelIn_;
      const int offsetOut = region * numChannelOut_;
      // count input objects
      int nTracks(0);
      int nStubs(0);
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
        const int index = offsetIn + channelIn;
        const int offset = index * numLayers_;
        const tt::StreamTrack& tracks = allTracks[index];
        nTracks += std::accumulate(tracks.begin(), tracks.end(), 0, validFrameT);
        for (int layer = 0; layer < numLayers_; layer++) {
          const tt::StreamStub& stubs = allStubs[offset + layer];
          nStubs += std::accumulate(stubs.begin(), stubs.end(), 0, validFrameS);
        }
      }
      // storage of input data
      std::vector<TrackKF> tracksKF;
      tracksKF.reserve(nTracks);
      std::vector<StubKF> stubsKF;
      stubsKF.reserve(nStubs);
      // h/w liked organized pointer to input data
      std::vector<std::vector<TrackKF*>> regionTracks(numChannelIn_);
      std::vector<std::vector<StubKF*>> regionStubs(numChannelIn_ * numLayers_);
      // read input data
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
        const int index = offsetIn + channelIn;
        const int offsetAll = index * numLayers_;
        const int offsetRegion = channelIn * numLayers_;
        const tt::StreamTrack& streamTrack = allTracks[index];
        std::vector<TrackKF*>& tracks = regionTracks[channelIn];
        tracks.reserve(streamTrack.size());
        for (const tt::FrameTrack& frame : streamTrack) {
          TrackKF* track = nullptr;
          if (frame.first.isNonnull()) {
            tracksKF.emplace_back(frame, dataFormats);
            track = &tracksKF.back();
          }
          tracks.push_back(track);
        }
        for (int layer = 0; layer < numLayers_; layer++) {
          for (const tt::FrameStub& frame : allStubs[offsetAll + layer]) {
            StubKF* stub = nullptr;
            if (frame.first.isNonnull()) {
              stubsKF.emplace_back(frame, dataFormats);
              stub = &stubsKF.back();
            }
            regionStubs[offsetRegion + layer].push_back(stub);
          }
        }
      }
      // empty storage of output data
      std::vector<TrackDR> tracksDR;
      tracksDR.reserve(nTracks);
      std::vector<StubDR> stubsDR;
      stubsDR.reserve(nStubs);
      // object to remove duplicates in a processing region
      DuplicateRemoval dr(setup, dataFormats, tracksDR, stubsDR);
      // empty h/w liked organized pointer to output data
      std::vector<std::vector<TrackDR*>> streamsTrack(numChannelOut_);
      std::vector<std::vector<StubDR*>> streamsStub(numChannelOut_ * numLayers_);
      // fill output data
      dr.produce(regionTracks, regionStubs, streamsTrack, streamsStub);
      // convert data to ed products
      for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
        const int index = offsetOut + channelOut;
        const int offsetRegion = channelOut * numLayers_;
        const int offsetAll = index * numLayers_;
        putT(streamsTrack[channelOut], acceptedTracks[index]);
        for (int layer = 0; layer < numLayers_; layer++)
          putS(streamsStub[offsetRegion + layer], acceptedStubs[offsetAll + layer]);
      }
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(acceptedStubs));
    iEvent.emplace(edPutTokenTracks_, std::move(acceptedTracks));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerDR);
