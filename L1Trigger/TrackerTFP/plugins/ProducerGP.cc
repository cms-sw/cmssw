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
    ~ProducerGP() override = default;

  private:
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

  void ProducerGP::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper classe to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty GP products
    tt::StreamsStub accepted(setup->numRegions() * dataFormats->numChannel(Process::gp));
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
    for (int region = 0; region < setup->numRegions(); region++) {
      const int offsetIn = region * dataFormats->numChannel(Process::pp);
      const int offsetOut = region * dataFormats->numChannel(Process::gp);
      // count input objects
      int nStubsPP(0);
      for (int channelIn = 0; channelIn < dataFormats->numChannel(Process::pp); channelIn++) {
        const tt::StreamStub& stream = streamsStub[offsetIn + channelIn];
        nStubsPP += std::accumulate(stream.begin(), stream.end(), 0, validFrame);
      }
      // storage of input data
      std::vector<StubPP> stubsPP;
      stubsPP.reserve(nStubsPP);
      // h/w liked organized pointer to input data
      std::vector<std::vector<StubPP*>> streamsIn(dataFormats->numChannel(Process::pp));
      // read input data
      for (int channelIn = 0; channelIn < dataFormats->numChannel(Process::pp); channelIn++) {
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
      std::vector<std::deque<StubGP*>> streamsOut(dataFormats->numChannel(Process::gp));
      // fill output data
      gp.produce(streamsIn, streamsOut);
      // convert data to ed products
      for (int channelOut = 0; channelOut < dataFormats->numChannel(Process::gp); channelOut++) {
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
