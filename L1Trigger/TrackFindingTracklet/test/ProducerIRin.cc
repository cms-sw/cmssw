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

using namespace std;
using namespace edm;
using namespace tt;

namespace trklet {

  /*! \class  trklet::ProducerIRin
   *  \brief  Extracts and rearranges StreamsStub from TTDTC
   *          Rearrangement may be configured to connect a reduced tracking chain to the correct L1 track board input channels.
   *  \author Thomas Schuh
   *  \date   2021, Oct
   */
  class ProducerIRin : public stream::EDProducer<> {
  public:
    explicit ProducerIRin(const ParameterSet&);
    ~ProducerIRin() override {}

  private:
    virtual void beginRun(const Run&, const EventSetup&) override;
    virtual void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}
    // ED input token of DTC Stubs
    EDGetTokenT<TTDTC> edGetTokenTTDTC_;
    // ED output token for stubs
    EDPutTokenT<StreamsStub> edPutTokenStubs_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // ChannelAssignment token
    ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to assign stubs to channel
    const ChannelAssignment* channelAssignment_;
  };

  ProducerIRin::ProducerIRin(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const InputTag& inputTag = iConfig.getParameter<InputTag>("InputTagDTC");
    const string& branchStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    // book in- and output ED products
    edGetTokenTTDTC_ = consumes<TTDTC>(inputTag);
    edPutTokenStubs_ = produces<StreamsStub>(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
    channelAssignment_ = nullptr;
  }

  void ProducerIRin::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
  }

  void ProducerIRin::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty IRin product
    StreamsStub streamStubs;
    // read in hybrid track finding product and produce KFin product
    if (setup_->configurationSupported()) {
      Handle<TTDTC> handleTTDTC;
      iEvent.getByToken<TTDTC>(edGetTokenTTDTC_, handleTTDTC);
      const vector<int>& channelEncoding = channelAssignment_->channelEncoding();
      const int numChannel = setup_->numRegions() * channelEncoding.size();
      streamStubs.reserve(numChannel);
      for (int tfpRegion : handleTTDTC->tfpRegions())
        for (int tfpChannel : channelEncoding)
          streamStubs.emplace_back(handleTTDTC->stream(tfpRegion, tfpChannel));
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, move(streamStubs));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerIRin);
