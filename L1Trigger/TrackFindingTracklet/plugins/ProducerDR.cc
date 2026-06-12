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

#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/DuplicateRemoval.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

namespace trklet {

  /*! \class  trklet::ProducerDR
   *  \brief  Emulates removal of duplicated TTTracks f/w.
   *          Track order determined by TrackMultiplexer affects performance
   *  \author Thomas Schuh
   *  \date   2023, Feb
   */
  class ProducerDR : public edm::stream::EDProducer<> {
  public:
    explicit ProducerDR(const edm::ParameterSet&);
    ~ProducerDR() override = default;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of Tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED input token of TTDTC
    edm::EDGetTokenT<TTDTC> edGetTokenDTC_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, trackerDTC::SetupRcd> esGetTokenDataFormats_;
  };

  ProducerDR::ProducerDR(const edm::ParameterSet& iConfig) {
    const std::string& label = iConfig.getParameter<std::string>("InputLabelDR");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    const edm::InputTag& inputTagDTC = iConfig.getParameter<edm::InputTag>("InputTagTTDTC");
    // book in- and output ED products
    edGetTokenTracks_ = consumes(edm::InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes(edm::InputTag(label, branchStubs));
    edGetTokenDTC_ = consumes(inputTagDTC);
    edPutTokenTracks_ = produces(branchTracks);
    edPutTokenStubs_ = produces(branchStubs);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
  }

  void ProducerDR::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // empty DR products
    tt::StreamsStub streamsStub(setup->sysNumRegion() * setup->drNumLayers());
    tt::StreamsTrack streamsTrack(setup->sysNumRegion());
    // read in TBout Product and produce KFin product
    const tt::StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    const TTDTC& ttDTC = iEvent.get(edGetTokenDTC_);
    for (int region = 0; region < setup->sysNumRegion(); region++) {
      // object to remove duplicated tracks in a processing region
      DuplicateRemoval dr(setup, dataFormats, region, ttDTC);
      // read in and organize input tracks and stubs
      dr.consume(tracks, stubs);
      // fill output products
      dr.produce(streamsTrack, streamsStub);
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerDR);
