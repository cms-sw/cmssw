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
#include "L1Trigger/TrackerTFP/interface/HoughTransform.h"

#include <string>
#include <vector>
#include <deque>
#include <numeric>
#include <algorithm>
#include <iterator>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerHT
   *  \brief  L1TrackTrigger Hough Transform emulator
   *  \author Thomas Schuh
   *  \date   2020, March
   */
  class ProducerHT : public edm::stream::EDProducer<> {
  public:
    explicit ProducerHT(const edm::ParameterSet&);
    ~ProducerHT() override = default;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of gp stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetToken_;
    // ED output token for accepted stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutToken_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    edm::ESGetToken<LayerEncoding, DataFormatsRcd> esGetTokenLayerEncoding_;
  };

  ProducerHT::ProducerHT(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelHT");
    const std::string& branch = iConfig.getParameter<std::string>("BranchStubs");
    // book in- and output ED products
    edGetToken_ = consumes<tt::StreamsStub>(edm::InputTag(label, branch));
    edPutToken_ = produces<tt::StreamsStub>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerHT::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    const LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty HT products
    tt::StreamsStub accepted(setup->numRegions() * dataFormats->numChannel(Process::ht));
    // read in DTC Product and produce TFP product
    const tt::StreamsStub& streamsStub = iEvent.get(edGetToken_);
    // helper
    auto validFrame = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto toFrame = [](StubHT* object) { return object ? object->frame() : tt::FrameStub(); };
    // produce HT output per region
    for (int region = 0; region < setup->numRegions(); region++) {
      const int offsetIn = region * dataFormats->numChannel(Process::gp);
      const int offsetOut = region * dataFormats->numChannel(Process::ht);
      // count input objects
      int nStubsGP(0);
      for (int channelIn = 0; channelIn < dataFormats->numChannel(Process::gp); channelIn++) {
        const tt::StreamStub& stream = streamsStub[offsetIn + channelIn];
        nStubsGP += std::accumulate(stream.begin(), stream.end(), 0, validFrame);
      }
      // storage of input data
      std::vector<StubGP> stubsGP;
      stubsGP.reserve(nStubsGP);
      // h/w liked organized pointer to input data
      std::vector<std::vector<StubGP*>> streamsIn(dataFormats->numChannel(Process::gp));
      // read input data
      for (int channelIn = 0; channelIn < dataFormats->numChannel(Process::gp); channelIn++) {
        const tt::StreamStub& streamStub = streamsStub[offsetIn + channelIn];
        std::vector<StubGP*>& stream = streamsIn[channelIn];
        stream.reserve(streamStub.size());
        for (const tt::FrameStub& frame : streamStub) {
          StubGP* stub = nullptr;
          if (frame.first.isNonnull()) {
            stubsGP.emplace_back(frame, dataFormats);
            stub = &stubsGP.back();
          }
          stream.push_back(stub);
        }
      }
      // container for output stubs
      std::vector<StubHT> stubsHT;
      // object to find initial rough candidates in r-phi in a region
      HoughTransform ht(setup, dataFormats, layerEncoding, stubsHT);
      // empty h/w liked organized pointer to output data
      std::vector<std::deque<StubHT*>> streamsOut(dataFormats->numChannel(Process::ht));
      // fill output data
      ht.produce(streamsIn, streamsOut);
      // convert data to ed products
      for (int channelOut = 0; channelOut < dataFormats->numChannel(Process::ht); channelOut++) {
        const std::deque<StubHT*>& objects = streamsOut[channelOut];
        tt::StreamStub& stream = accepted[offsetOut + channelOut];
        stream.reserve(objects.size());
        std::transform(objects.begin(), objects.end(), std::back_inserter(stream), toFrame);
      }
    }
    // store products
    iEvent.emplace(edPutToken_, std::move(accepted));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerHT);
