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
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

namespace trklet {

  /*! \class  trklet::ProducerIRin
   *  \brief  Transforms TTTDCinto f/w comparable format for summer chain configuratiotn
   *  \author Thomas Schuh
   *  \date   2021, Oct
   */
  class ProducerIRin : public edm::stream::EDProducer<> {
  public:
    explicit ProducerIRin(const edm::ParameterSet&);
    ~ProducerIRin() override {}

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() {}
    // ED input token of DTC Stubs
    edm::EDGetTokenT<TTDTC> edGetTokenTTDTC_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // ChannelAssignment token
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // configuration
    edm::ParameterSet iConfig_;
    // helper class to store configurations
    const tt::Setup* setup_;
    // helper class to assign stubs to channel
    const ChannelAssignment* channelAssignment_;
    // map of used tfp channels
    std::vector<int> channelEncoding_;
  };

  ProducerIRin::ProducerIRin(const edm::ParameterSet& iConfig) : iConfig_(iConfig) {
    const edm::InputTag& inputTag = iConfig.getParameter<edm::InputTag>("InputTagDTC");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubsAccepted");
    // book in- and output ED products
    edGetTokenTTDTC_ = consumes<TTDTC>(inputTag);
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes<tt::Setup, tt::SetupRcd, edm::Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<ChannelAssignment, ChannelAssignmentRcd, edm::Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
  }

  void ProducerIRin::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    channelAssignment_ = const_cast<ChannelAssignment*>(&iSetup.getData(esGetTokenChannelAssignment_));
    // map of used tfp channels
    channelEncoding_ = channelAssignment_->channelEncoding();
  }

  void ProducerIRin::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // empty IRin product
    tt::StreamsStub streamStubs;
    // read in hybrid track finding product and produce KFin product
    edm::Handle<TTDTC> handleTTDTC;
    iEvent.getByToken<TTDTC>(edGetTokenTTDTC_, handleTTDTC);
    const int numChannel = channelEncoding_.size();
    streamStubs.reserve(numChannel);
    for (int tfpRegion : handleTTDTC->tfpRegions())
      for (int tfpChannel : channelEncoding_)
        streamStubs.emplace_back(handleTTDTC->stream(tfpRegion, tfpChannel));
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(streamStubs));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerIRin);
