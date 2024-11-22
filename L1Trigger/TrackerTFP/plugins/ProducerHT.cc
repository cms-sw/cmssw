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

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerHT
   *  \brief  L1TrackTrigger Hough Transform emulator
   *  \author Thomas Schuh
   *  \date   2020, March
   */
  class ProducerHT : public stream::EDProducer<> {
  public:
    explicit ProducerHT(const ParameterSet&);
    ~ProducerHT() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}
    // ED input token of gp stubs
    EDGetTokenT<StreamsStub> edGetToken_;
    // ED output token for accepted stubs
    EDPutTokenT<StreamsStub> edPutToken_;
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
    //
    const LayerEncoding* layerEncoding_ = nullptr;
  };

  ProducerHT::ProducerHT(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("InputLabelHT");
    const string& branch = iConfig.getParameter<string>("BranchStubs");
    // book in- and output ED products
    edGetToken_ = consumes<StreamsStub>(InputTag(label, branch));
    edPutToken_ = produces<StreamsStub>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
  }

  void ProducerHT::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    //
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
  }

  void ProducerHT::produce(Event& iEvent, const EventSetup& iSetup) {
    static const int numChannelIn = dataFormats_->numChannel(Process::gp);
    static const int numChannelOut = dataFormats_->numChannel(Process::ht);
    static const int numRegions = setup_->numRegions();
    // empty HT products
    StreamsStub accepted(numRegions * numChannelOut);
    // read in DTC Product and produce TFP product
    Handle<StreamsStub> handle;
    iEvent.getByToken<StreamsStub>(edGetToken_, handle);
    const StreamsStub& streamsStub = *handle.product();
    // helper
    auto validFrame = [](int& sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto toFrame = [](StubHT* object) { return object ? object->frame() : FrameStub(); };
    // produce HT output per region
    for (int region = 0; region < numRegions; region++) {
      const int offsetIn = region * numChannelIn;
      const int offsetOut = region * numChannelOut;
      // count input objects
      int nStubsGP(0);
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const StreamStub& stream = streamsStub[offsetIn + channelIn];
        nStubsGP += accumulate(stream.begin(), stream.end(), 0, validFrame);
      }
      // storage of input data
      vector<StubGP> stubsGP;
      stubsGP.reserve(nStubsGP);
      // h/w liked organized pointer to input data
      vector<vector<StubGP*>> streamsIn(numChannelIn);
      // read input data
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const StreamStub& streamStub = streamsStub[offsetIn + channelIn];
        vector<StubGP*>& stream = streamsIn[channelIn];
        stream.reserve(streamStub.size());
        for (const FrameStub& frame : streamStub) {
          StubGP* stub = nullptr;
          if (frame.first.isNonnull()) {
            stubsGP.emplace_back(frame, dataFormats_);
            stub = &stubsGP.back();
          }
          stream.push_back(stub);
        }
      }
      // container for output stubs
      vector<StubHT> stubsHT;
      // object to find initial rough candidates in r-phi in a region
      HoughTransform ht(iConfig_, setup_, dataFormats_, layerEncoding_, stubsHT);
      // empty h/w liked organized pointer to output data
      vector<deque<StubHT*>> streamsOut(numChannelOut);
      // fill output data
      ht.produce(streamsIn, streamsOut);
      // convert data to ed products
      for (int channelOut = 0; channelOut < numChannelOut; channelOut++) {
        const deque<StubHT*>& objects = streamsOut[channelOut];
        StreamStub& stream = accepted[offsetOut + channelOut];
        stream.reserve(objects.size());
        transform(objects.begin(), objects.end(), back_inserter(stream), toFrame);
      }
    }
    // store products
    iEvent.emplace(edPutToken_, move(accepted));
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerHT);
