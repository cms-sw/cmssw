#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

namespace trklet {

  /*! \class  trklet::ProducerFakeDR
   *  \brief  tranforms tracklet TTTracks into KF emulator input format
   *  \author Thomas Schuh
   *  \date   2025, July
   */
  class ProducerFakeDR : public edm::stream::EDProducer<> {
  public:
    explicit ProducerFakeDR(const edm::ParameterSet&);
    ~ProducerFakeDR() override = default;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    // ED input token of TTTracks
    edm::EDGetTokenT<tt::TTTracks> edGetTokenTracks_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, ChannelAssignmentRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    edm::ESGetToken<trackerTFP::LayerEncoding, trackerTFP::DataFormatsRcd> esGetTokenLayerEncoding_;
  };

  ProducerFakeDR::ProducerFakeDR(const edm::ParameterSet& iConfig) {
    const edm::InputTag& inputTag = iConfig.getParameter<edm::InputTag>("InputTagTracklet");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetTokenTracks_ = consumes<tt::TTTracks>(inputTag);
    edPutTokenStubs_ = produces<tt::StreamsStub>(branchStubs);
    edPutTokenTracks_ = produces<tt::StreamsTrack>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes();
    esGetTokenDataFormats_ = esConsumes();
    esGetTokenLayerEncoding_ = esConsumes();
  }

  void ProducerFakeDR::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer id
    const trackerTFP::LayerEncoding* layerEncoding = &iSetup.getData(esGetTokenLayerEncoding_);
    // empty output products
    std::vector<int> sizes(setup->numRegions(), 0);
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : iEvent.get(edGetTokenTracks_))
      sizes[ttTrack.phiSector()]++;
    tt::StreamsTrack streamsTrack(setup->numRegions());
    for (int iRegion = 0; iRegion < setup->numRegions(); iRegion++)
      streamsTrack.reserve(sizes[iRegion]);
    tt::StreamsStub streamsStub(setup->numRegions() * setup->numLayers());
    for (int iRegion = 0; iRegion < setup->numRegions(); iRegion++) {
      const int offset = iRegion * setup->numLayers();
      const int size = sizes[iRegion];
      for (int iLayer = 0; iLayer < setup->numLayers(); iLayer++)
        streamsStub[offset + iLayer].reserve(size);
    }
    // process TTTracks
    edm::Handle<tt::TTTracks> handle;
    iEvent.getByToken(edGetTokenTracks_, handle);
    for (int iTrack = 0; iTrack < static_cast<int>(handle->size()); iTrack++) {
      const TTTrackRef ttTrackRef(handle, iTrack);
      const int iRegion = ttTrackRef->phiSector();
      // track parameter
      const double inv2R = -.5 * ttTrackRef->rInv();
      const double phiT =
          tt::deltaPhi(ttTrackRef->phi() + inv2R * setup->chosenRofPhi() - iRegion * setup->baseRegion());
      const double cot = ttTrackRef->tanL();
      const double zT = ttTrackRef->z0() + cot * setup->chosenRofZ();
      // range checks
      const bool validInv2R = dataFormats->format(Variable::inv2R, Process::dr).inRange(inv2R);
      const bool validPhiT = dataFormats->format(Variable::phiT, Process::dr).inRange(phiT);
      const bool validZT = dataFormats->format(Variable::zT, Process::dr).inRange(zT);
      if (!validInv2R || !validPhiT || !validZT)
        continue;
      // track parameter shifts to adjust stubs
      const double dinv2R = inv2R - dataFormats->format(Variable::inv2R, Process::dr).digi(inv2R);
      const double dphiT = phiT - dataFormats->format(Variable::phiT, Process::dr).digi(phiT);
      const double dcot = cot - dataFormats->format(Variable::zT, Process::dr).digi(zT) / setup->chosenRofZ();
      const double dzT = zT - dataFormats->format(Variable::zT, Process::dr).digi(zT);
      // process stubs
      const int offset = iRegion * setup->numLayers();
      const std::vector<int>& le = layerEncoding->layerEncoding(zT);
      TTBV hitPattern(0, setup->numLayers());
      for (const TTStubRef& ttStubRef : ttTrackRef->getStubRefs()) {
        // layer encoding
        const int layerId = setup->layerId(ttStubRef);
        const int iLayer = std::distance(le.begin(), std::find(le.begin(), le.end(), layerId));
        if (hitPattern.test(iLayer))
          continue;
        // stub parameter
        tt::SensorModule* sm = setup->sensorModule(ttStubRef);
        const GlobalPoint gp = setup->stubPos(ttStubRef);
        const double r = gp.perp() - setup->chosenRofPhi();
        const double rZ = gp.perp() - setup->chosenRofZ();
        double phi = tt::deltaPhi(gp.phi() - iRegion * setup->baseRegion() - phiT - r * inv2R);
        double z = gp.z() - zT - rZ * cot;
        const double dZ = .5 * sm->dZ();
        const double dPhi = .5 * sm->dPhi(inv2R);
        // linear correction
        const double d = inv2R * gp.perp();
        const double cor = std::asin(d) - d;
        phi -= cor;
        z -= cor / inv2R * cot;
        // shift stubs accoring to track shifts
        phi += dphiT + r * dinv2R;
        z += dzT + rZ * dcot;
        // range checks
        const bool validR = dataFormats->format(Variable::r, Process::dr).inRange(r);
        const bool validPhi = dataFormats->format(Variable::phi, Process::dr).inRange(phi);
        const bool validZ = dataFormats->format(Variable::z, Process::dr).inRange(z);
        const bool validDPhi = dataFormats->format(Variable::dPhi, Process::dr).inRange(dPhi);
        const bool validDZ = dataFormats->format(Variable::dZ, Process::dr).inRange(dZ);
        if (!validR || !validPhi || !validZ || !validDPhi || !validDZ)
          continue;
        // store stub
        hitPattern.set(iLayer);
        const StubDR stubDR(ttStubRef, dataFormats, r, phi, z, dPhi, dZ);
        streamsStub[offset + iLayer].push_back(stubDR.frame());
      }
      // check enough stubs
      if (hitPattern.count() < setup->kfMinLayers()) {
        for (int iLayer : hitPattern.ids())
          streamsStub[offset + iLayer].pop_back();
        continue;
      }
      // fill empty layer
      for (int iLayer : hitPattern.ids(false))
        streamsStub[offset + iLayer].emplace_back(tt::FrameStub());
      // store track
      const TrackDR trackDR(ttTrackRef, dataFormats, inv2R, phiT, zT);
      streamsTrack[iRegion].push_back(trackDR.frame());
    }
    // store products
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerFakeDR);
