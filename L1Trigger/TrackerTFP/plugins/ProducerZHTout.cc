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

#include <string>
#include <vector>
#include <deque>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerZHTout
   *  \brief  transforms SF output into TTTracks
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerZHTout : public stream::EDProducer<> {
  public:
    explicit ProducerZHTout(const ParameterSet&);
    ~ProducerZHTout() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of sf stubs
    EDGetTokenT<StreamsStub> edGetToken_;
    // ED output token of TTTracks
    EDPutTokenT<vector<TTTrack<Ref_Phase2TrackerDigi_>>> edPutToken_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_ = nullptr;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_ = nullptr;
  };

  ProducerZHTout::ProducerZHTout(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const string& label = iConfig.getParameter<string>("LabelZHT");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    // book in- and output ED products
    edGetToken_ = consumes<StreamsStub>(InputTag(label, branchAcceptedStubs));
    edPutToken_ = produces<vector<TTTrack<Ref_Phase2TrackerDigi_>>>(branchAcceptedTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
  }

  void ProducerZHTout::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
  }

  void ProducerZHTout::produce(Event& iEvent, const EventSetup& iSetup) {
    const DataFormat& dfCot = dataFormats_->format(Variable::cot, Process::zht);
    const DataFormat& dfZT = dataFormats_->format(Variable::zT, Process::zht);
    const DataFormat& dfPhiT = dataFormats_->format(Variable::phiT, Process::zht);
    const DataFormat& dfinv2R = dataFormats_->format(Variable::inv2R, Process::zht);
    // empty SFout product
    deque<TTTrack<Ref_Phase2TrackerDigi_>> ttTracks;
    // read in SF Product and produce SFout product
    if (setup_->configurationSupported()) {
      Handle<StreamsStub> handle;
      iEvent.getByToken<StreamsStub>(edGetToken_, handle);
      const StreamsStub& streams = *handle.product();
      for (int region = 0; region < setup_->numRegions(); region++) {
        for (int channel = 0; channel < dataFormats_->numChannel(Process::zht); channel++) {
          const int index = region * dataFormats_->numChannel(Process::zht) + channel;
          // convert stream to stubs
          const StreamStub& stream = streams[index];
          vector<StubZHT> stubs;
          stubs.reserve(stream.size());
          for (const FrameStub& frame : stream)
            if (frame.first.isNonnull())
              stubs.emplace_back(frame, dataFormats_);
          // form tracks
          int i(0);
          for (auto it = stubs.begin(); it != stubs.end();) {
            const auto start = it;
            const int id = it->trackId();
            auto different = [id](const StubZHT& stub) { return id != stub.trackId(); };
            it = find_if(it, stubs.end(), different);
            vector<TTStubRef> ttStubRefs;
            ttStubRefs.reserve(distance(start, it));
            transform(start, it, back_inserter(ttStubRefs), [](const StubZHT& stub) { return stub.ttStubRef(); });
            const double zT = dfZT.floating(start->zT());
            const double cot = dfCot.floating(start->cot());
            const double phiT = dfPhiT.floating(start->phiT());
            const double inv2R = dfinv2R.floating(start->inv2R());
            ttTracks.emplace_back(inv2R, phiT, cot, zT, 0., 0., 0., 0., 0., 0, 0, 0.);
            ttTracks.back().setStubRefs(ttStubRefs);
            ttTracks.back().setPhiSector(start->sectorPhi() + region * setup_->numSectorsPhi());
            ttTracks.back().setEtaSector(start->sectorEta());
            ttTracks.back().setTrackSeedType(start->trackId());
            if (i++ == setup_->zhtMaxTracks())
              break;
          }
        }
      }
    }
    // store product
    iEvent.emplace(edPutToken_, ttTracks.begin(), ttTracks.end());
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerZHTout);