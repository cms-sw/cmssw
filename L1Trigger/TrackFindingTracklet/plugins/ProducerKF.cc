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
#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/KalmanFilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"

#include <string>
#include <vector>

namespace trklet {

  /*! \class  trklet::ProducerKF
   *  \brief  L1TrackTrigger Kamlan Filter emulator
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerKF : public edm::stream::EDProducer<> {
  public:
    explicit ProducerKF(const edm::ParameterSet&);
    ~ProducerKF() override = default;

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override {
      std::stringstream ss;
      if (printDebug_)
        kalmanFilterFormats_.endJob(ss);
      edm::LogPrint(moduleDescription().moduleName()) << ss.str();
    }
    // call old KF
    void oldKF(const tt::StreamsTrack&, const tt::StreamsStub&, tt::StreamsTrack&, tt::StreamsStub&);
    // ED input token of sf stubs and tracks
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // Setup token
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    // DataFormats token
    edm::ESGetToken<DataFormats, trackerDTC::SetupRcd> esGetTokenDataFormats_;
    // helper class to extract structured data from tt::Frames
    const DataFormats* dataFormats_;
    // provides dataformats of Kalman filter internals
    KalmanFilterFormats kalmanFilterFormats_;
    //
    ConfigKF iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    //
    tmtt::Settings settings_;
    //
    tmtt::KFParamsComb tmtt_;
    // print end job internal unused MSB
    bool printDebug_;
  };

  ProducerKF::ProducerKF(const edm::ParameterSet& iConfig) : settings_(iConfig), tmtt_(&settings_, 4, "KF4ParamsComb") {
    iConfig_.enableIntegerEmulation_ = iConfig.getParameter<bool>("EnableIntegerEmulation");
    iConfig_.widthR00_ = iConfig.getParameter<int>("WidthR00");
    iConfig_.widthR11_ = iConfig.getParameter<int>("WidthR11");
    iConfig_.widthC00_ = iConfig.getParameter<int>("WidthC00");
    iConfig_.widthC01_ = iConfig.getParameter<int>("WidthC01");
    iConfig_.widthC11_ = iConfig.getParameter<int>("WidthC11");
    iConfig_.widthC22_ = iConfig.getParameter<int>("WidthC22");
    iConfig_.widthC23_ = iConfig.getParameter<int>("WidthC23");
    iConfig_.widthC33_ = iConfig.getParameter<int>("WidthC33");
    iConfig_.baseShiftx0_ = iConfig.getParameter<int>("BaseShiftx0");
    iConfig_.baseShiftx1_ = iConfig.getParameter<int>("BaseShiftx1");
    iConfig_.baseShiftx2_ = iConfig.getParameter<int>("BaseShiftx2");
    iConfig_.baseShiftx3_ = iConfig.getParameter<int>("BaseShiftx3");
    iConfig_.baseShiftr0_ = iConfig.getParameter<int>("BaseShiftr0");
    iConfig_.baseShiftr1_ = iConfig.getParameter<int>("BaseShiftr1");
    iConfig_.baseShiftS00_ = iConfig.getParameter<int>("BaseShiftS00");
    iConfig_.baseShiftS01_ = iConfig.getParameter<int>("BaseShiftS01");
    iConfig_.baseShiftS12_ = iConfig.getParameter<int>("BaseShiftS12");
    iConfig_.baseShiftS13_ = iConfig.getParameter<int>("BaseShiftS13");
    iConfig_.baseShiftR00_ = iConfig.getParameter<int>("BaseShiftR00");
    iConfig_.baseShiftR11_ = iConfig.getParameter<int>("BaseShiftR11");
    iConfig_.baseShiftInvR00Approx_ = iConfig.getParameter<int>("BaseShiftInvR00Approx");
    iConfig_.baseShiftInvR11Approx_ = iConfig.getParameter<int>("BaseShiftInvR11Approx");
    iConfig_.baseShiftInvR00Cor_ = iConfig.getParameter<int>("BaseShiftInvR00Cor");
    iConfig_.baseShiftInvR11Cor_ = iConfig.getParameter<int>("BaseShiftInvR11Cor");
    iConfig_.baseShiftInvR00_ = iConfig.getParameter<int>("BaseShiftInvR00");
    iConfig_.baseShiftInvR11_ = iConfig.getParameter<int>("BaseShiftInvR11");
    iConfig_.baseShiftS00Shifted_ = iConfig.getParameter<int>("BaseShiftS00Shifted");
    iConfig_.baseShiftS01Shifted_ = iConfig.getParameter<int>("BaseShiftS01Shifted");
    iConfig_.baseShiftS12Shifted_ = iConfig.getParameter<int>("BaseShiftS12Shifted");
    iConfig_.baseShiftS13Shifted_ = iConfig.getParameter<int>("BaseShiftS13Shifted");
    iConfig_.baseShiftK00_ = iConfig.getParameter<int>("BaseShiftK00");
    iConfig_.baseShiftK10_ = iConfig.getParameter<int>("BaseShiftK10");
    iConfig_.baseShiftK21_ = iConfig.getParameter<int>("BaseShiftK21");
    iConfig_.baseShiftK31_ = iConfig.getParameter<int>("BaseShiftK31");
    iConfig_.baseShiftC00_ = iConfig.getParameter<int>("BaseShiftC00");
    iConfig_.baseShiftC01_ = iConfig.getParameter<int>("BaseShiftC01");
    iConfig_.baseShiftC11_ = iConfig.getParameter<int>("BaseShiftC11");
    iConfig_.baseShiftC22_ = iConfig.getParameter<int>("BaseShiftC22");
    iConfig_.baseShiftC23_ = iConfig.getParameter<int>("BaseShiftC23");
    iConfig_.baseShiftC33_ = iConfig.getParameter<int>("BaseShiftC33");
    iConfig_.baseShiftInvDH_ = iConfig.getParameter<int>("BaseShiftInvDH");
    iConfig_.baseShiftInvDH2_ = iConfig.getParameter<int>("BaseShiftInvDH2");
    iConfig_.baseShiftHv0_ = iConfig.getParameter<int>("BaseShiftHv0");
    iConfig_.baseShiftHv1_ = iConfig.getParameter<int>("BaseShiftHv1");
    iConfig_.baseShiftH2v0_ = iConfig.getParameter<int>("BaseShiftH2v0");
    iConfig_.baseShiftH2v1_ = iConfig.getParameter<int>("BaseShiftH2v1");
    printDebug_ = iConfig.getParameter<bool>("PrintKFDebug");
    const std::string& label = iConfig.getParameter<std::string>("InputLabelKF");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes(edm::InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes(edm::InputTag(label, branchTracks));
    edPutTokenStubs_ = produces(branchStubs);
    edPutTokenTracks_ = produces(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<edm::Transition::BeginRun>();
  }

  void ProducerKF::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    settings_.setMagneticField(setup_->sysBField());
    // helper class to extract structured data from tt::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    kalmanFilterFormats_.consume(dataFormats_, iConfig_);
  }

  void ProducerKF::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // empty KF products
    tt::StreamsStub streamsStub(setup_->sysNumRegion() * setup_->kfNumLayers());
    tt::StreamsTrack streamsTrack(setup_->sysNumRegion());
    // read in DR Product and produce KF product
    const tt::StreamsStub& stubs = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& tracks = iEvent.get(edGetTokenTracks_);
    if (setup_->kfUseSimulation())
      oldKF(tracks, stubs, streamsTrack, streamsStub);
    else {
      for (int region = 0; region < setup_->sysNumRegion(); region++) {
        // object to fit tracks in a processing region
        KalmanFilter kf(setup_, dataFormats_, &kalmanFilterFormats_, region);
        // read in and organize input tracks and stubs
        kf.consume(tracks, stubs);
        // fill output products
        kf.produce(streamsStub, streamsTrack);
      }
    }
    // store products
    iEvent.emplace(edPutTokenStubs_, std::move(streamsStub));
    iEvent.emplace(edPutTokenTracks_, std::move(streamsTrack));
  }

  // call old KF
  void ProducerKF::oldKF(const tt::StreamsTrack& tracksIn,
                         const tt::StreamsStub& stubsIn,
                         tt::StreamsTrack& tracksOut,
                         tt::StreamsStub& stubsOut) {
    std::vector<double> zTs;
    zTs.reserve(settings_.etaRegions().size());
    for (double eta : settings_.etaRegions())
      zTs.emplace_back(std::sinh(eta) * settings_.chosenRofZ());
    for (int region = 0; region < setup_->sysNumRegion(); region++) {
      const double phiR = region * setup_->regRangePhiT();
      const int offsetIn = region * setup_->drNumLayers();
      const int offsetOut = region * setup_->kfNumLayers();
      const tt::StreamTrack& streamTrack = tracksIn[region];
      const int sizeT = streamTrack.size();
      tracksOut[region].reserve(sizeT);
      for (int layer = 0; layer < setup_->kfNumLayers(); layer++)
        stubsOut[offsetOut + layer].reserve(sizeT);
      for (int iFrame = 0; iFrame < sizeT; iFrame++) {
        const tt::FrameTrack& frameTrack = streamTrack[iFrame];
        const TTTrackRef& ttTrackRef = frameTrack.first;
        if (ttTrackRef.isNull())
          continue;
        TrackDR trackDR(frameTrack, dataFormats_);
        // collect stubs
        std::vector<StubDR> stubsDR;
        stubsDR.reserve(setup_->drNumLayers());
        for (int layer = 0; layer < setup_->drNumLayers(); layer++) {
          const tt::FrameStub& frameStub = stubsIn[offsetIn + layer][iFrame];
          if (frameStub.first.isNonnull())
            stubsDR.emplace_back(frameStub, dataFormats_);
        }
        // convert stubs
        std::vector<tmtt::Stub> stubs;
        stubs.reserve(stubsDR.size());
        std::vector<tmtt::Stub*> stubsPtr;
        stubsPtr.reserve(stubsDR.size());
        for (const StubDR& stubDR : stubsDR) {
          const TTStubRef& ttStubRef = stubDR.frame().first;
          const trackerDTC::SensorModule* sm = setup_->sensorModule(ttStubRef);
          stubs.emplace_back(ttStubRef,
                             stubDR.r(),
                             tt::deltaPhi(stubDR.phi() + phiR),
                             stubDR.z(),
                             sm->layerId(),
                             sm->layerIdReduced(),
                             sm->pitchRow(),
                             sm->pitchCol(),
                             sm->psModule(),
                             sm->barrel(),
                             sm->tilted());
          stubsPtr.push_back(&stubs.back());
        }
        // convert intput Track
        const double zT = ttTrackRef->z0() + settings_.chosenRofZ() * ttTrackRef->tanL();
        int iEtaReg = 0;
        for (; iEtaReg < 15; iEtaReg++)
          if (zT < zTs[iEtaReg + 1])
            break;
        const tmtt::L1track3D l1track3D(&settings_,
                                        stubsPtr,
                                        .5 * ttTrackRef->rInv() / setup_->sysInvPtToDphi(),
                                        ttTrackRef->phi(),
                                        ttTrackRef->z0(),
                                        ttTrackRef->tanL(),
                                        0.,
                                        region,
                                        iEtaReg);
        // perform fit
        const tmtt::L1fittedTrack trackFitted(tmtt_.fit(l1track3D));
        if (!trackFitted.accepted())
          continue;
        // convert to output track
        const double inv2R = -trackFitted.qOverPt() * setup_->sysInvPtToDphi();
        const double phi0 = tt::deltaPhi(trackFitted.phi0() - phiR);
        const double cot = trackFitted.tanLambda();
        const double z0 = trackFitted.z0();
        const TrackKF trackKF(trackDR, inv2R, phi0, cot, z0);
        // convert to output stubs
        const int sizeS = trackFitted.stubs().size();
        std::vector<StubKF> stubsKF;
        stubsKF.reserve(sizeS);
        for (tmtt::Stub* stub : trackFitted.stubs()) {
          const TTStubRef& ttStubRef = stub->ttStubRef();
          const trackerDTC::SensorModule* sm = setup_->sensorModule(ttStubRef);
          auto same = [&ttStubRef](const StubDR& s) { return s.frame().first == ttStubRef; };
          const auto it = std::find_if(stubsDR.begin(), stubsDR.end(), same);
          const double phi = it->phi() - it->r() * inv2R - phi0;
          const double z = it->z() - it->r() * cot - z0;
          stubsKF.emplace_back(*it, sm->layerIdReduced(), it->r(), phi, z, it->dPhi(), it->dZ());
        }
        // store
        tracksOut[region].push_back(trackKF.frame());
        for (int layer = 0; layer < sizeS; layer++)
          stubsOut[offsetOut + layer].push_back(stubsKF[layer].frame());
        for (int layer = sizeS; layer < setup_->kfNumLayers(); layer++)
          stubsOut[offsetOut + layer].emplace_back();
      }
    }
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::ProducerKF);
