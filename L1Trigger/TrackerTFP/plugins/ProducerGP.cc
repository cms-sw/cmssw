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

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerGP
   *  \brief  L1TrackTrigger Geometric Processor emulator
   *  \author Thomas Schuh
   *  \date   2020, March
   */
  class ProducerGP : public stream::EDProducer<> {
  public:
    explicit ProducerGP(const ParameterSet&);
    ~ProducerGP() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}
    // ED input token of pp objects
    EDGetTokenT<StreamsStub> edGetToken_;
    // ED output token for accepted objects
    EDPutTokenT<StreamsStub> edPutToken_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // configuration
    ParameterSet iConfig_;
    // helper classe to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
    // helper class to encode layer
    const LayerEncoding* layerEncoding_ = nullptr;
  };

  ProducerGP::ProducerGP(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelGP");
    const string& branch = iConfig.getParameter<string>("BranchStubs");
    // book in- and output ED products
    edGetToken_ = consumes<StreamsStub>(InputTag(label, branch));
    edPutToken_ = produces<StreamsStub>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
  }

  void ProducerGP::beginRun(const Run& iRun, const EventSetup& iSetup) {
    setup_ = &iSetup.getData(esGetTokenSetup_);
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
  }

  void ProducerGP::produce(Event& iEvent, const EventSetup& iSetup) {
    static const int numChannelIn = dataFormats_->numChannel(Process::pp);
    static const int numChannelOut = dataFormats_->numChannel(Process::gp);
    static const int numRegions = setup_->numRegions();
    // empty GP products
    StreamsStub accepted(numRegions * numChannelOut);
    // read in DTC Product and produce TFP product
    Handle<StreamsStub> handle;
    iEvent.getByToken<StreamsStub>(edGetToken_, handle);
    const StreamsStub& streamsStub = *handle.product();
    // helper
    auto validFrame = [](int sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto nSectors = [](int sum, const StubPP& object) {
      const int nPhiT = object.phiTMax() - object.phiTMin() + 1;
      const int nZT = object.zTMax() - object.zTMin() + 1;
      return sum += nPhiT * nZT;
    };
    auto toFrame = [](StubGP* object) { return object ? object->frame() : FrameStub(); };
    // produce GP product per region
    for (int region = 0; region < numRegions; region++) {
      const int offsetIn = region * numChannelIn;
      const int offsetOut = region * numChannelOut;
      // count input objects
      int nStubsPP(0);
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const StreamStub& stream = streamsStub[offsetIn + channelIn];
        nStubsPP += accumulate(stream.begin(), stream.end(), 0, validFrame);
      }
      // storage of input data
      vector<StubPP> stubsPP;
      stubsPP.reserve(nStubsPP);
      // h/w liked organized pointer to input data
      vector<vector<StubPP*>> streamsIn(numChannelIn);
      // read input data
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const StreamStub& streamStub = streamsStub[offsetIn + channelIn];
        vector<StubPP*>& stream = streamsIn[channelIn];
        stream.reserve(streamStub.size());
        for (const FrameStub& frame : streamStub) {
          StubPP* stubPP = nullptr;
          if (frame.first.isNonnull()) {
            stubsPP.emplace_back(frame, dataFormats_);
            stubPP = &stubsPP.back();
          }
          stream.push_back(stubPP);
        }
      }
      // predict upper limit of GP stubs
      const int nStubsGP = accumulate(stubsPP.begin(), stubsPP.end(), 0, nSectors);
      // container of GP stubs
      vector<StubGP> stubsGP;
      stubsGP.reserve(nStubsGP);
      // object to route Stubs of one region to one stream per sector
      GeometricProcessor gp(iConfig_, setup_, dataFormats_, layerEncoding_, stubsGP);
      // empty h/w liked organized pointer to output data
      vector<deque<StubGP*>> streamsOut(numChannelOut);
      // fill output data
      gp.produce(streamsIn, streamsOut);
      // convert data to ed products
      for (int channelOut = 0; channelOut < numChannelOut; channelOut++) {
        const deque<StubGP*>& objects = streamsOut[channelOut];
        StreamStub& stream = accepted[offsetOut + channelOut];
        stream.reserve(objects.size());
        transform(objects.begin(), objects.end(), back_inserter(stream), toFrame);
      }
    }
    // store products
    iEvent.emplace(edPutToken_, std::move(accepted));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerGP);
