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
    edGetTokenTracks_ = consumes(inputTag);
    edPutTokenStubs_ = produces(branchStubs);
    edPutTokenTracks_ = produces(branchTracks);
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
      const double phiR = iRegion * setup->baseRegion();
      // track parameter
      const double d0 = -ttTrackRef->d0();
      double inv2R = -.5 * ttTrackRef->rInv();
      double phi0 = tt::deltaPhi(ttTrackRef->phi() - phiR);
      double cot = ttTrackRef->tanL();
      double z0 = ttTrackRef->z0();
      double R = .5 / inv2R;
      double R0 = R + d0;
      double phiT = phi0 + std::asin((setup->chosenRofPhi() * setup->chosenRofPhi() + R0 * R0 - R * R) / 2. /
                                     setup->chosenRofPhi() / R0);
      double zT = z0 + std::abs(R) * cot *
                           std::acos((R * R + R0 * R0 - setup->chosenRofZ() * setup->chosenRofZ()) / 2. / R / R0);
      // range checks
      const bool validInv2R = dataFormats->format(Variable::inv2R, Process::dr).isCovered(inv2R);
      const bool validPhiT = dataFormats->format(Variable::phiT, Process::dr).isCovered(phiT);
      const bool validZT = dataFormats->format(Variable::zT, Process::dr).isCovered(zT);
      if (!validInv2R || !validPhiT || !validZT)
        continue;
      // digitised track parameter
      inv2R = dataFormats->format(Variable::inv2R, Process::tm).digi(inv2R);
      phiT = dataFormats->format(Variable::phiT, Process::tm).digi(phiT);
      cot = dataFormats->format(Variable::zT, Process::tm).digi(zT) / setup->chosenRofZ();
      zT = dataFormats->format(Variable::zT, Process::tm).digi(zT);
      R = .5 / inv2R;
      R0 = R + d0;
      phi0 = phiT - std::asin((setup->chosenRofPhi() * setup->chosenRofPhi() + R0 * R0 - R * R) / 2. /
                              setup->chosenRofPhi() / R0);
      z0 = zT -
           std::abs(R) * cot * std::acos((R * R + R0 * R0 - setup->chosenRofZ() * setup->chosenRofZ()) / 2. / R / R0);
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
        const double r = gp.perp();
        const double rPhi = gp.perp() - setup->chosenRofPhi();
        const double trackPhi = phi0 + std::asin((r * r + R0 * R0 - R * R) / 2. / r / R0);
        const double trackZ = z0 + std::abs(R) * cot * std::acos((R * R + R0 * R0 - r * r) / 2. / R / R0);
        double phi = tt::deltaPhi(gp.phi() - phiR - trackPhi);
        double z = gp.z() - trackZ;
        const double dZ = .5 * sm->dZ(cot);
        const double dPhi = .5 * sm->dPhi(inv2R);
        // range checks
        const bool validR = dataFormats->format(Variable::r, Process::dr).isCovered(rPhi);
        const bool validPhi = dataFormats->format(Variable::phi, Process::dr).isCovered(phi);
        const bool validZ = dataFormats->format(Variable::z, Process::dr).isCovered(z);
        const bool validDPhi = dataFormats->format(Variable::dPhi, Process::dr).isCovered(dPhi);
        const bool validDZ = dataFormats->format(Variable::dZ, Process::dr).isCovered(dZ);
        if (!validR || !validPhi || !validZ || !validDPhi || !validDZ)
          continue;
        // store stub
        hitPattern.set(iLayer);
        const StubDR stubDR(ttStubRef, dataFormats, rPhi, phi, z, dPhi, dZ);
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
