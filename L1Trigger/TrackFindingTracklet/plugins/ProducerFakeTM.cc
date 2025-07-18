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
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include <vector>
#include <deque>
#include <set>
#include <iterator>
#include <cmath>
#include <numeric>

namespace trklet {

  /*! \class  trklet::ProducerFakeTM
   *  \brief  tranforms tracklet TTTracks into KF emulator input format
   *  \author Thomas Schuh
   *  \date   2025, July
   */
  class ProducerFakeTM : public edm::stream::EDProducer<> {
  public:
    explicit ProducerFakeTM(const edm::ParameterSet&);
    ~ProducerFakeTM() override = default;

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
    // ChannelAssignment token
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
  };

  ProducerFakeTM::ProducerFakeTM(const edm::ParameterSet& iConfig) {
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
    esGetTokenChannelAssignment_ = esConsumes();
  }

  void ProducerFakeTM::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment = &iSetup.getData(esGetTokenChannelAssignment_);
    // empty output products and prep
    std::vector<std::set<TTStubRef>> ttStubRefs(setup->numRegions() * channelAssignment->tmNumLayers());
    std::vector<int> sizes(setup->numRegions(), 0);
    for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : iEvent.get(edGetTokenTracks_)) {
      const int offset = ttTrack.phiSector() * channelAssignment->tmNumLayers();
      sizes[ttTrack.phiSector()]++;
      for (const TTStubRef& ttStubRef : ttTrack.getStubRefs())
        ttStubRefs[offset + setup->trackletLayerId(ttStubRef)].insert(ttStubRef);
    }
    tt::StreamsTrack streamsTrack(setup->numRegions());
    for (int iRegion = 0; iRegion < setup->numRegions(); iRegion++)
      streamsTrack.reserve(sizes[iRegion]);
    tt::StreamsStub streamsStub(setup->numRegions() * channelAssignment->tmNumLayers());
    for (int iRegion = 0; iRegion < setup->numRegions(); iRegion++) {
      const int offset = iRegion * channelAssignment->tmNumLayers();
      const int size = sizes[iRegion];
      for (int iLayer = 0; iLayer < channelAssignment->tmNumLayers(); iLayer++)
        streamsStub[offset + iLayer].reserve(size);
    }
    // process TTTracks
    edm::Handle<tt::TTTracks> handle;
    iEvent.getByToken(edGetTokenTracks_, handle);
    std::vector<std::pair<TrackTM, std::vector<StubTM*>>> tracks;
    tracks.reserve(handle->size());
    std::deque<StubTM> stubsTM;
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
      const bool validInv2R = dataFormats->format(Variable::inv2R, Process::tm).inRange(inv2R);
      const bool validPhiT = dataFormats->format(Variable::phiT, Process::tm).inRange(phiT);
      const bool validZT = dataFormats->format(Variable::zT, Process::tm).inRange(zT);
      if (!validInv2R || !validPhiT || !validZT)
        continue;
      const TrackTM trackTM(ttTrackRef, dataFormats, inv2R, phiT, zT);
      std::vector<StubTM*> stubs(channelAssignment->tmNumLayers(), nullptr);
      // track parameter shifts to adjust stubs
      const double dinv2R = inv2R - dataFormats->format(Variable::inv2R, Process::tm).digi(inv2R);
      const double dphiT = phiT - dataFormats->format(Variable::phiT, Process::tm).digi(phiT);
      const double dcot = cot - dataFormats->format(Variable::zT, Process::tm).digi(zT) / setup->chosenRofZ();
      const double dzT = zT - dataFormats->format(Variable::zT, Process::tm).digi(zT);
      // process stubs
      const int offset = iRegion * channelAssignment->tmNumLayers();
      TTBV hitPattern(0, channelAssignment->tmNumLayers());
      for (const TTStubRef& ttStubRef : ttTrackRef->getStubRefs()) {
        const int iLayer = setup->trackletLayerId(ttStubRef);
        if (hitPattern.test(iLayer))
          continue;
        // stub parameter
        const std::set<TTStubRef>& stubIds = ttStubRefs[offset + iLayer];
        tt::SensorModule* sm = setup->sensorModule(ttStubRef);
        const GlobalPoint gp = setup->stubPos(ttStubRef);
        const int stubId = std::distance(stubIds.begin(), std::find(stubIds.begin(), stubIds.end(), ttStubRef));
        const bool pst = (sm->barrel() && sm->tilt()) || (!sm->barrel() && sm->psModule());
        const double r = gp.perp() - setup->chosenRofPhi();
        const double rZ = gp.perp() - setup->chosenRofZ();
        double phi = tt::deltaPhi(gp.phi() - iRegion * setup->baseRegion() - phiT - r * inv2R);
        double z = gp.z() - zT - rZ * cot;
        // linear correction
        const double d = inv2R * gp.perp();
        const double cor = std::asin(d) - d;
        phi -= cor;
        z -= cor / inv2R * cot;
        // shift stubs accoring to track shifts
        phi += dphiT + r * dinv2R;
        z += dzT + rZ * dcot;
        // range checks
        const bool validR = dataFormats->format(Variable::r, Process::tm).inRange(r);
        const bool validPhi = dataFormats->format(Variable::phi, Process::tm).inRange(phi);
        const bool validZ = dataFormats->format(Variable::z, Process::tm).inRange(z);
        if (!validR || !validPhi || !validZ)
          continue;
        // store stub
        hitPattern.set(iLayer);
        stubsTM.emplace_back(ttStubRef, dataFormats, 2 * stubId + (pst ? 1 : 0), r, phi, z);
        stubs[iLayer] = &stubsTM.back();
      }
      // check enough stubs
      const bool validTrack = hitPattern.count() >= setup->kfMinLayers();
      if (validTrack)
        tracks.emplace_back(trackTM, stubs);
    }
    // sort tracks by seed type
    const std::vector<int>& order = channelAssignment->tmMuxOrder();
    auto seedValue = [&order](const auto& p) {
      const int seedType = p.first.frame().first->trackSeedType();
      return std::distance(order.begin(), std::find(order.begin(), order.end(), seedType));
    };
    auto smaller = [seedValue](const auto& lhs, const auto& rhs) { return seedValue(lhs) < seedValue(rhs); };
    std::sort(tracks.begin(), tracks.end(), smaller);
    for (const std::pair<TrackTM, std::vector<StubTM*>>& track : tracks) {
      const int iRegion = track.first.frame().first->phiSector();
      const int offset = iRegion * channelAssignment->tmNumLayers();
      streamsTrack[iRegion].emplace_back(track.first.frame());
      for (int iLayer = 0; iLayer < channelAssignment->tmNumLayers(); iLayer++) {
        StubTM* stub = track.second[iLayer];
        streamsStub[offset + iLayer].push_back(stub ? stub->frame() : tt::FrameStub());
      }
    }
    // store products
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerFakeTM);
