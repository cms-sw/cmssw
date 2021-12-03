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
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

using namespace std;
using namespace edm;
using namespace trackerTFP;
using namespace tt;

namespace trklet {

  /*! \class  trklet::ProducerKFin
   *  \brief  Transforms format of TTTracks from Tracklet pattern reco. to that expected by KF input.
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class ProducerKFin : public stream::EDProducer<> {
  public:
    explicit ProducerKFin(const ParameterSet&);
    ~ProducerKFin() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of TTTracks
    EDGetTokenT<TTTracks> edGetTokenTTTracks_;
    // ED output token for stubs
    EDPutTokenT<StreamsStub> edPutTokenAcceptedStubs_;
    EDPutTokenT<StreamsStub> edPutTokenLostStubs_;
    // ED output token for tracks
    EDPutTokenT<StreamsTrack> edPutTokenAcceptedTracks_;
    EDPutTokenT<StreamsTrack> edPutTokenLostTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // ChannelAssignment token
    ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to extract structured data from TTDTC::Frames
    const DataFormats* dataFormats_;
    // helper class to encode layer
    const LayerEncoding* layerEncoding_;
    // helper class to assign tracks to channel
    ChannelAssignment* channelAssignment_;
    //
    bool enableTruncation_;
  };

  ProducerKFin::ProducerKFin(const ParameterSet& iConfig) : iConfig_(iConfig) {
    const InputTag& inputTag = iConfig.getParameter<InputTag>("InputTag");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLostStubs = iConfig.getParameter<string>("BranchLostStubs");
    const string& branchLostTracks = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenTTTracks_ = consumes<TTTracks>(inputTag);
    edPutTokenAcceptedStubs_ = produces<StreamsStub>(branchAcceptedStubs);
    edPutTokenAcceptedTracks_ = produces<StreamsTrack>(branchAcceptedTracks);
    edPutTokenLostStubs_ = produces<StreamsStub>(branchLostStubs);
    edPutTokenLostTracks_ = produces<StreamsTrack>(branchLostTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<ChannelAssignment, ChannelAssignmentRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
    dataFormats_ = nullptr;
    layerEncoding_ = nullptr;
    channelAssignment_ = nullptr;
    //
    enableTruncation_ = iConfig.getParameter<bool>("EnableTruncation");
  }

  void ProducerKFin::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from TTDTC::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    // helper class to assign tracks to channel
    channelAssignment_ = const_cast<ChannelAssignment*>(&iSetup.getData(esGetTokenChannelAssignment_));
  }

  void ProducerKFin::produce(Event& iEvent, const EventSetup& iSetup) {
    // dataformat used for track cotTheta wrt eta sector centre
    const DataFormat& dfcot = dataFormats_->format(Variable::cot, Process::kfin);
    // dataformat used for track z at raiud chosenRofZ wrt eta sector centre
    const DataFormat& dfzT = dataFormats_->format(Variable::zT, Process::kfin);
    // dataformat used for track inv2R in 1 / cm
    const DataFormat& dfinv2R = dataFormats_->format(Variable::inv2R, Process::kfin);
    // dataformat used for track phi at radius schoenRofPhi wrt phi sector centre
    const DataFormat& dfphiT = dataFormats_->format(Variable::phiT, Process::kfin);
    const int numStreamsTracks = setup_->numRegions() * channelAssignment_->numChannels();
    const int numStreamsStubs = numStreamsTracks * setup_->numLayers();
    // empty KFin products
    StreamsStub streamAcceptedStubs(numStreamsStubs);
    StreamsTrack streamAcceptedTracks(numStreamsTracks);
    StreamsStub streamLostStubs(numStreamsStubs);
    StreamsTrack streamLostTracks(numStreamsTracks);
    // read in hybrid track finding product and produce KFin product
    if (setup_->configurationSupported()) {
      // create TTrackRefs
      Handle<TTTracks> handleTTTracks;
      iEvent.getByToken<TTTracks>(edGetTokenTTTracks_, handleTTTracks);
      vector<TTTrackRef> ttTrackRefs;
      ttTrackRefs.reserve(handleTTTracks->size());
      for (int i = 0; i < (int)handleTTTracks->size(); i++)
        ttTrackRefs.emplace_back(TTTrackRef(handleTTTracks, i));
      // Assign input tracks to channels according to TrackBuilder step.
      vector<vector<TTTrackRef>> ttTrackRefsStreams(numStreamsTracks);
      vector<int> nTTTracksStreams(numStreamsTracks, 0);
      int channelId;
      for (const TTTrackRef& ttTrackRef : ttTrackRefs)
        if (channelAssignment_->channelId(ttTrackRef, channelId))
          nTTTracksStreams[channelId]++;
      channelId = 0;
      for (int nTTTracksStream : nTTTracksStreams)
        ttTrackRefsStreams[channelId++].reserve(nTTTracksStream);
      for (const TTTrackRef& ttTrackRef : ttTrackRefs)
        if (channelAssignment_->channelId(ttTrackRef, channelId))
          ttTrackRefsStreams[channelId].push_back(ttTrackRef);
      for (channelId = 0; channelId < numStreamsTracks; channelId++) {
        // Create vector of stubs/tracks in KF format from TTTracks
        deque<FrameTrack> streamTracks;
        vector<deque<FrameStub>> streamsStubs(setup_->numLayers());
        for (const TTTrackRef& ttTrackRef : ttTrackRefsStreams[channelId]) {
          // get rz parameter
          const double cotGlobal = ttTrackRef->tanL();
          const double zTGlobal = ttTrackRef->z0() + setup_->chosenRofZ() * cotGlobal;
          int binEta(-1);
          for (; binEta < setup_->numSectorsEta(); binEta++)
            if (zTGlobal < sinh(setup_->boundarieEta(binEta + 1)) * setup_->chosenRofZ())
              break;
          // cut on outer eta sector boundaries
          if (binEta == -1 || binEta == setup_->numSectorsEta())
            continue;
          const double cot = dfcot.digi(cotGlobal - setup_->sectorCot(binEta));
          const double zT = dfzT.digi(zTGlobal - setup_->sectorCot(binEta) * setup_->chosenRofZ());
          // cut on eta and |z0| < 15 cm
          if (!dfzT.inRange(zT) || !dfcot.inRange(cot))
            continue;
          const int binZT = dfzT.toUnsigned(dfzT.integer(zT));
          const int binCot = dfcot.toUnsigned(dfcot.integer(cot));
          // get set of kf layers for this rough r-z track parameter
          const vector<int>& le = layerEncoding_->layerEncoding(binEta, binZT, binCot);
          // get rphi parameter
          double inv2R = dfinv2R.digi(-ttTrackRef->rInv() / 2.);
          // calculcate track phi at radius hybridChosenRofPhi with respect to phi sector centre
          double phiT = dfphiT.digi(deltaPhi(ttTrackRef->phi() + dataFormats_->chosenRofPhi() * inv2R -
                                             ttTrackRef->phiSector() * setup_->baseRegion()));
          const int sectorPhi = phiT < 0. ? 0 : 1;  // dirty hack
          phiT -= (sectorPhi - .5) * setup_->baseSector();
          // cut on nonant size and pt
          if (!dfphiT.inRange(phiT) || !dfinv2R.inRange(inv2R))
            continue;
          const double offsetPhi =
              (ttTrackRef->phiSector() * setup_->numSectorsPhi() + sectorPhi - .5) * setup_->baseSector();
          // check hitPattern
          TTBV hitPattern(0, setup_->numLayers());
          static constexpr double scalePhi = 4.0;
          static constexpr double scaleZ = 2.0;
          for (const TTStubRef& ttStubRef : ttTrackRef->getStubRefs()) {
            const GlobalPoint& gp = setup_->stubPos(ttStubRef);
            const double rphi = gp.perp() - dataFormats_->chosenRofPhi();
            const double rz = gp.perp() - setup_->chosenRofZ();
            const double phi = deltaPhi(gp.phi() - offsetPhi - (phiT + inv2R * rphi));
            const double offsetZ = setup_->sectorCot(binEta) * gp.perp();
            const double z = gp.z() - offsetZ - (zT + cot * rz);
            const double dPhi = setup_->dPhi(ttStubRef, inv2R);
            const double dZ = setup_->dZ(ttStubRef, cotGlobal);
            const double rangePhi = abs(rphi) * dfinv2R.base() * scalePhi + dfphiT.base() * scalePhi + dPhi;
            const double rangeZ = abs(rz) * dfcot.base() * scaleZ + dfzT.base() * scaleZ + dZ;
            // cut on phi and z residuals
            if (abs(phi) > rangePhi / 2. || abs(z) > rangeZ / 2.)
              continue;
            // layers consitent with rough r-z track parameters are counted from 0 onwards
            int layer = distance(le.begin(), find(le.begin(), le.end(), setup_->layerId(ttStubRef)));
            // put stubs from layer 7 to layer 6 since layer 7 almost never has stubs
            if (layer >= setup_->numLayers())
              layer = setup_->numLayers() - 1;
            hitPattern.set(layer);
          }
          //cout << hitPattern << endl;
          if (hitPattern.count() < setup_->kfMinLayers())
            continue;
          // create Stubs
          vector<int> layerCounts(setup_->numLayers(), 0);
          for (const TTStubRef& ttStubRef : ttTrackRef->getStubRefs()) {
            const GlobalPoint& gp = setup_->stubPos(ttStubRef);
            const double rphi = gp.perp() - dataFormats_->chosenRofPhi();
            const double rz = gp.perp() - setup_->chosenRofZ();
            const double phi = deltaPhi(gp.phi() - offsetPhi - (phiT + inv2R * rphi));
            const double offsetZ = setup_->sectorCot(binEta) * gp.perp();
            const double z = gp.z() - offsetZ - (zT + cot * rz);
            const double dPhi = setup_->dPhi(ttStubRef, inv2R);
            const double dZ = setup_->dZ(ttStubRef, cotGlobal);
            const double rangePhi = abs(rphi) * dfinv2R.base() * scalePhi + dfphiT.base() * scalePhi + dPhi;
            const double rangeZ = abs(rz) * dfcot.base() * scaleZ + dfzT.base() * scaleZ + dZ;
            // cut on phi and z residuals
            if (abs(phi) > rangePhi / 2. || abs(z) > rangeZ / 2.)
              continue;
            // layers consitent with rough r-z track parameters are counted from 0 onwards
            int layer = distance(le.begin(), find(le.begin(), le.end(), setup_->layerId(ttStubRef)));
            // put stubs from layer 7 to layer 6 since layer 7 almost never has stubs
            if (layer >= setup_->numLayers())
              layer = setup_->numLayers() - 1;
            // cut on max 4 stubs per layer
            if (layerCounts[layer] >= setup_->zhtMaxStubsPerLayer())
              continue;
            layerCounts[layer]++;
            const StubKFin stubKFin(ttStubRef, dataFormats_, rphi, phi, z, dPhi, dZ, layer);
            streamsStubs[layer].emplace_back(stubKFin.frame());
          }
          const int size = *max_element(layerCounts.begin(), layerCounts.end());
          for (int layer = 0; layer < setup_->numLayers(); layer++) {
            deque<FrameStub>& stubs = streamsStubs[layer];
            const int nGaps = size - layerCounts[layer];
            stubs.insert(stubs.end(), nGaps, FrameStub());
          }
          const TTBV& maybePattern = layerEncoding_->maybePattern(binEta, binZT, binCot);
          const TrackKFin track(ttTrackRef, dataFormats_, maybePattern, phiT, inv2R, zT, cot, sectorPhi, binEta);
          streamTracks.emplace_back(track.frame());
          const int nGaps = size - 1;
          streamTracks.insert(streamTracks.end(), nGaps, FrameTrack());
        }
        // transform deques to vectors & emulate truncation
        auto limitTracks = next(streamTracks.begin(), min(setup_->numFrames(), (int)streamTracks.size()));
        if (!enableTruncation_)
          limitTracks = streamTracks.end();
        streamAcceptedTracks[channelId] = StreamTrack(streamTracks.begin(), limitTracks);
        streamLostTracks[channelId] = StreamTrack(limitTracks, streamTracks.end());
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          const int index = channelId * setup_->numLayers() + layer;
          deque<FrameStub>& stubs = streamsStubs[layer];
          auto limitStubs = next(stubs.begin(), min(setup_->numFrames(), (int)stubs.size()));
          if (!enableTruncation_)
            limitStubs = stubs.end();
          streamAcceptedStubs[index] = StreamStub(stubs.begin(), limitStubs);
          streamLostStubs[index] = StreamStub(limitStubs, stubs.end());
        }
      }
    }
    // store products
    iEvent.emplace(edPutTokenAcceptedStubs_, move(streamAcceptedStubs));
    iEvent.emplace(edPutTokenAcceptedTracks_, move(streamAcceptedTracks));
    iEvent.emplace(edPutTokenLostStubs_, move(streamLostStubs));
    iEvent.emplace(edPutTokenLostTracks_, move(streamLostTracks));
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerKFin);