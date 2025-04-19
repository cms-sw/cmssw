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
#include "L1Trigger/TrackerTFP/interface/GeometricProcessor.h"

#include <string>
#include <vector>
#include <deque>
#include <numeric>
#include <algorithm>
#include <iterator>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerGP
   *  \brief  L1TrackTrigger Geometric Processor emulator
   *  \author Thomas Schuh
   *  \date   2020, March
   */
  class ProducerGP : public edm::stream::EDProducer<> {
  public:
    explicit ProducerGP(const edm::ParameterSet&);
    ~ProducerGP() override {}

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of pp objects
    edm::EDGetTokenT<tt::StreamsStub> edGetToken_;
    // ED output token for accepted objects
    edm::EDPutTokenT<tt::StreamsStub> edPutToken_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    edm::ESGetToken<LayerEncoding, DataFormatsRcd> esGetTokenLayerEncoding_;
    // number of input channel
    int numChannelIn_;
    // number of output channel
    int numChannelOut_;
    // number of processing regions
    int numRegions_;
  };

  ProducerGP::ProducerGP(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelGP");
    const std::string& branch = iConfig.getParameter<std::string>("BranchStubs");
    // book in- and output ED products
    edGetToken_ = consumes<tt::StreamsStub>(edm::InputTag(label, branch));
    edPutToken_ = produces<tt::StreamsStub>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerGP::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    // helper classe to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    numRegions_ = setup->numRegions();
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    numChannelIn_ = dataFormats->numChannel(Process::pp);
    numChannelOut_ = dataFormats->numChannel(Process::gp);
  }

  void ProducerGP::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper classe to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty GP products
    tt::StreamsStub accepted(numRegions_ * numChannelOut_);
    // read in DTC Product and produce TFP product
    const tt::StreamsStub& streamsStub = iEvent.get(edGetToken_);
    // helper
    auto validFrame = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto nSectors = [](int sum, const StubPP& object) {
      const int nPhiT = object.phiTMax() - object.phiTMin() + 1;
      const int nZT = object.zTMax() - object.zTMin() + 1;
      return sum + nPhiT * nZT;
    };
    auto toFrame = [](StubGP* object) { return object ? object->frame() : tt::FrameStub(); };
    // produce GP product per region
    for (int region = 0; region < numRegions_; region++) {
      const int offsetIn = region * numChannelIn_;
      const int offsetOut = region * numChannelOut_;
      // count input objects
      int nStubsPP(0);
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
        const tt::StreamStub& stream = streamsStub[offsetIn + channelIn];
        nStubsPP += std::accumulate(stream.begin(), stream.end(), 0, validFrame);
      }
      // storage of input data
      std::vector<StubPP> stubsPP;
      stubsPP.reserve(nStubsPP);
      // h/w liked organized pointer to input data
      std::vector<std::vector<StubPP*>> streamsIn(numChannelIn_);
      // read input data
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
        const tt::StreamStub& streamStub = streamsStub[offsetIn + channelIn];
        std::vector<StubPP*>& stream = streamsIn[channelIn];
        stream.reserve(streamStub.size());
        for (const tt::FrameStub& frame : streamStub) {
          StubPP* stubPP = nullptr;
          if (frame.first.isNonnull()) {
            stubsPP.emplace_back(frame, dataFormats);
            stubPP = &stubsPP.back();
          }
          stream.push_back(stubPP);
        }
      }
      // predict upper limit of GP stubs
      const int nStubsGP = std::accumulate(stubsPP.begin(), stubsPP.end(), 0, nSectors);
      // container of GP stubs
      std::vector<StubGP> stubsGP;
      stubsGP.reserve(nStubsGP);
      // object to route Stubs of one region to one stream per sector
      GeometricProcessor gp(setup, dataFormats, layerEncoding, stubsGP);
      // empty h/w liked organized pointer to output data
      std::vector<std::deque<StubGP*>> streamsOut(numChannelOut_);
      // fill output data
      gp.produce(streamsIn, streamsOut);
      // convert data to ed products
      for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
        const std::deque<StubGP*>& objects = streamsOut[channelOut];
        tt::StreamStub& stream = accepted[offsetOut + channelOut];
        stream.reserve(objects.size());
        std::transform(objects.begin(), objects.end(), std::back_inserter(stream), toFrame);
      }
    }
    // store products
    iEvent.emplace(edPutToken_, std::move(accepted));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerGP);
