#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackMultiplexer.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"

#include <TProfile2D.h>
#include <TH1F.h>

#include <vector>
#include <deque>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>

namespace trklet {

  /*! \class  trklet::AnalyzerTB
   *  \brief  Class to analyze emulated Track Builder 
   *  \author Thomas Schuh
   *  \date   2025, Nov
   */
  class AnalyzerTB : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    AnalyzerTB(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    struct StubTB {
      StubTB(const TTStubRef& ttStubRef, double r, double z, double inv2R, bool seed = false)
          : ttStubRef_(ttStubRef), seed_(seed), r_(r), z_(z) {}
      TTStubRef ttStubRef_;
      bool seed_;
      double r_;
      double z_;
    };
    struct TrackTB {
      TrackTB(const TTTrackRef& ttTrackRef,
              int seedType,
              double inv2R,
              double cot,
              double z0,
              const std::deque<StubTB*>& stubs)
          : ttTrackRef_(ttTrackRef), seedType_(seedType), inv2R_(inv2R), cot_(cot), z0_(z0), stubs_(stubs) {}
      TTTrackRef ttTrackRef_;
      int seedType_;
      double inv2R_;
      double cot_;
      double z0_;
      std::deque<StubTB*> stubs_;
    };
    // truncates double precision of val into base precision
    double digi(double val, double base) const { return (tt::floor(val / base) + .5) * base; }
    // read in tracks and stubs
    void consume(const tt::StreamsTrack&, const tt::StreamsStub&, std::deque<TrackTB>&, std::deque<StubTB>&) const;
    // ED input token of Tracks
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracks_;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubs_;
    // ED output token for stubs
    edm::EDPutTokenT<tt::StreamsStub> edPutTokenStubs_;
    // ED output token for tracks
    edm::EDPutTokenT<tt::StreamsTrack> edPutTokenTracks_;
    // ED output token for stub association for tracking efficiency
    edm::EDGetTokenT<tt::StubAssociation> edGetTokenEff_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokensetup;
    // ChannelAssignment token
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetTokenChannelAssignment_;
    // Associator token
    edm::ESGetToken<tt::Associator, tt::SetupRcd> esGetTokenAssociator_;
    // helper class to store configurations
    const tt::Setup* setup_;
    // helper class to assign tracks to channel
    const ChannelAssignment* channelAssignment_;
    // helper class to store tracklet configurations
    Settings settings_;
    //
    double baseUinv2R_;
    double baseUphiT_;
    double baseUcot_;
    double baseUzT_;
    double baseUr_;
    double baseUphi_;
    double baseUz_;
    double baseInvCot_;
    double baseScot_;
    //
    TH1F* his_;
    std::vector<TH1F*> hisST_;
    TProfile2D* prof_;
    std::vector<TProfile2D*> profST_;
    TH1F* hisTT_;
    std::vector<TH1F*> hisTTST_;
    TProfile2D* profTT_;
    std::vector<TProfile2D*> profTTST_;
    TH1F* hisCot_;
    TH1F* hisZ0_;
    std::vector<TH1F*> hisCotST_;
    std::vector<TH1F*> hisZ0ST_;
  };

  AnalyzerTB::AnalyzerTB(const edm::ParameterSet& iConfig) {
    usesResource("TFileService");
    const std::string& label = iConfig.getParameter<std::string>("InputLabelTM");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    const std::string& labelMC = iConfig.getParameter<std::string>("LabelMC");
    const std::string& branchEff = iConfig.getParameter<std::string>("BranchEff");
    // book in- and output ED products
    edGetTokenTracks_ = consumes(edm::InputTag(label, branchTracks));
    edGetTokenStubs_ = consumes(edm::InputTag(label, branchStubs));
    edGetTokenEff_ = consumes(edm::InputTag(labelMC, branchEff));
    // book ES products
    esGetTokensetup = esConsumes<edm::Transition::BeginRun>();
    esGetTokenChannelAssignment_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenAssociator_ = esConsumes();
    //
    baseUinv2R_ = .5 * settings_.kphi1() / settings_.kr() * pow(2, settings_.rinv_shift());
    baseUphiT_ = settings_.kphi1() * pow(2, settings_.phi0_shift());
    baseUcot_ = settings_.kz() / settings_.kr() * pow(2, settings_.t_shift());
    baseUzT_ = settings_.kz() * pow(2, settings_.z0_shift());
    baseUr_ = settings_.kr();
    baseUphi_ = settings_.kphi1();
    baseUz_ = settings_.kz();
  }

  void AnalyzerTB::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokensetup);
    // helper class to assign tracks to channel
    channelAssignment_ = &iSetup.getData(esGetTokenChannelAssignment_);
    //
    const int baseShiftInvCot = tt::ceil(std::log2(setup_->tbMaxR() / setup_->tbMinZ())) - setup_->widthDSPbu();
    baseInvCot_ = std::pow(2, baseShiftInvCot);
    const int unusedMSBScot =
        tt::floor(std::log2(baseUcot_ * std::pow(2.0, channelAssignment_->tmWidthCot()) / 2. / setup_->maxCot()));
    const int baseShiftScot = channelAssignment_->tmWidthCot() - unusedMSBScot - 1 - setup_->widthAddrBRAM18();
    baseScot_ = baseUcot_ * std::pow(2.0, baseShiftScot);
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    // stub z postion from tracklet residuals plus tracklet seed parameter vs stub z psotion from TTStubs projected to stub radius from tracklet using tracklet seed parameter
    dir = fs->mkdir("TB track parameter");
    his_ = dir.make<TH1F>("His z residual", ";", 128, -2., 2.);
    hisST_ = std::vector<TH1F*>(8);
    for (int st = 0; st < 8; st++)
      hisST_[st] = dir.make<TH1F>(("His z residual Seed Type " + std::to_string(st)).c_str(), ";", 128, -2., 2.);
    prof_ = dir.make<TProfile2D>("prof z residual", ";", 512, -300, 300., 128, 0., 120.);
    profST_ = std::vector<TProfile2D*>(8);
    for (int st = 0; st < 8; st++)
      profST_[st] = dir.make<TProfile2D>(
          ("prof z residual Seed Type " + std::to_string(st)).c_str(), ";", 512, -300, 300., 128, 0., 120.);
    // stub z postion from tracklet residuals plus recalculated seed parameter vs stub z psotion from TTStubs projected to stub radius from tracklet using recalculated seed parameter
    dir = fs->mkdir("TTStub track parameter");
    hisTT_ = dir.make<TH1F>("His z residual", ";", 128, -2., 2.);
    hisTTST_ = std::vector<TH1F*>(8);
    for (int st = 0; st < 8; st++)
      hisTTST_[st] = dir.make<TH1F>(("His z residual Seed Type " + std::to_string(st)).c_str(), ";", 128, -2., 2.);
    profTT_ = dir.make<TProfile2D>("prof z residual", ";", 512, -300, 300., 128, 0., 120.);
    profTTST_ = std::vector<TProfile2D*>(8);
    for (int st = 0; st < 8; st++)
      profTTST_[st] = dir.make<TProfile2D>(
          ("prof z residual Seed Type " + std::to_string(st)).c_str(), ";", 512, -300, 300., 128, 0., 120.);
    // Helix parameters from tracklet vs helix parameter calculated from seed TTSTubs
    dir = fs->mkdir("TB vs TTStub track parameter");
    hisCot_ = dir.make<TH1F>("His cot residual", ";", 128, -.2, .2);
    hisZ0_ = dir.make<TH1F>("His z0 residual", ";", 128, -2., 2.);
    hisCotST_ = std::vector<TH1F*>(8);
    hisZ0ST_ = std::vector<TH1F*>(8);
    for (int st = 0; st < 8; st++) {
      hisCotST_[st] = dir.make<TH1F>(("His cot residual Seed Type " + std::to_string(st)).c_str(), ";", 128, -.2, .2);
      hisZ0ST_[st] = dir.make<TH1F>(("His z0 residual Seed Type " + std::to_string(st)).c_str(), ";", 128, -2., 2.);
    }
  }

  void AnalyzerTB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    tt::Associator associator = iSetup.getData(esGetTokenAssociator_);
    associator.consume(iEvent.get(edGetTokenEff_));
    // read in TBout Product and produce TM product
    const tt::StreamsStub& streamsStub = iEvent.get(edGetTokenStubs_);
    const tt::StreamsTrack& streamsTrack = iEvent.get(edGetTokenTracks_);
    // storage for tracks and stubs
    std::deque<StubTB> tbStubs;
    std::deque<TrackTB> tbTracks;
    // read in tracks and stubs
    consume(streamsTrack, streamsStub, tbTracks, tbStubs);
    // analyze in tracks and stubs
    for (const TrackTB& trackTB : tbTracks) {
      // Do reco-truth matching.
      std::vector<TTStubRef> ttStubRefs;
      ttStubRefs.reserve(trackTB.stubs_.size());
      for (StubTB* stubTB : trackTB.stubs_)
        ttStubRefs.push_back(stubTB->ttStubRef_);
      const std::vector<TPPtr>& tpPtrs = associator.associateFinal(ttStubRefs);
      // analyze stub z residuals
      for (StubTB* stubTB : trackTB.stubs_) {
        // calculate TBStub z position
        const double z =
            trackTB.z0_ + trackTB.cot_ / trackTB.inv2R_ * std::asin(stubTB->r_ * trackTB.inv2R_) + stubTB->z_;
        const GlobalPoint gp = setup_->stubPos(stubTB->ttStubRef_);
        // calculate TTStub z position at TBStub radius
        const double ttZ =
            gp.z() - trackTB.cot_ / trackTB.inv2R_ * std::asin((gp.perp() - stubTB->r_) * trackTB.inv2R_);
        // caluclate error
        const double delta = ttZ - z;
        // fill stub z error
        his_->Fill(delta);
        hisST_[trackTB.seedType_]->Fill(delta);
        prof_->Fill(gp.z(), gp.perp(), std::abs(delta));
        profST_[trackTB.seedType_]->Fill(gp.z(), gp.perp(), std::abs(delta));
      }
      // recalcualte seed track parameter
      std::vector<double> r, z;
      for (StubTB* stubTB : trackTB.stubs_) {
        if (!stubTB->seed_)
          continue;
        const GlobalPoint gp = setup_->stubPos(stubTB->ttStubRef_);
        r.push_back(gp.perp());
        z.push_back(gp.z());
      }
      if (r[0] > r[1]) {
        std::reverse(r.begin(), r.end());
        std::reverse(z.begin(), z.end());
      }
      const double invDR = 1. / (r[1] - r[0]);
      const double cot = (z[1] - z[0]) * invDR;
      const double z0 = (r[1] * z[0] - r[0] * z[1]) * invDR;
      // fill seed track parameter error
      hisCot_->Fill(trackTB.cot_ - cot);
      hisZ0_->Fill(trackTB.z0_ - z0);
      hisCotST_[trackTB.seedType_]->Fill(trackTB.cot_ - cot);
      hisZ0ST_[trackTB.seedType_]->Fill(trackTB.z0_ - z0);
      // repeat stub z error study using recalculated seed track parameter
      for (StubTB* stubTB : trackTB.stubs_) {
        // origianl TBStub z residual
        double z = stubTB->z_;
        // shift z residual corresponding to parameter shift
        if (!stubTB->seed_)
          z += (trackTB.cot_ - cot) / trackTB.inv2R_ * std::asin(stubTB->r_ * trackTB.inv2R_) + (trackTB.z0_ - z0);
        // calculate z position
        z += z0 + cot / trackTB.inv2R_ * std::asin(stubTB->r_ * trackTB.inv2R_);
        const GlobalPoint gp = setup_->stubPos(stubTB->ttStubRef_);
        // calculate TTStub z position at TBStub radius
        const double ttZ = gp.z() - cot / trackTB.inv2R_ * std::asin((gp.perp() - stubTB->r_) * trackTB.inv2R_);
        // caluclate error
        const double delta = ttZ - z;
        // fill stub z error
        hisTT_->Fill(delta);
        hisTTST_[trackTB.seedType_]->Fill(delta);
        profTT_->Fill(gp.z(), gp.perp(), std::abs(delta));
        profTTST_[trackTB.seedType_]->Fill(gp.z(), gp.perp(), std::abs(delta));
      }
    }
  }

  // read in tracks and stubs
  void AnalyzerTB::consume(const tt::StreamsTrack& streamsTrack,
                           const tt::StreamsStub& streamsStub,
                           std::deque<TrackTB>& tbTracks,
                           std::deque<StubTB>& tbStubs) const {
    for (int region = 0; region < setup_->numRegions(); region++) {
      const int offsetTrack = region * channelAssignment_->numChannelsTrack();
      for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
        const int numP = channelAssignment_->numProjectionLayers(channel);
        const int channelTrack = offsetTrack + channel;
        const int offsetStub = channelAssignment_->offsetStub(channelTrack);
        const tt::StreamTrack& streamTrack = streamsTrack[channelTrack];
        for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
          const TTTrackRef& ttTrackRef = streamTrack[frame].first;
          if (ttTrackRef.isNull())
            continue;
          //convert track parameter
          double inv2R = digi(-ttTrackRef->rInv() / 2., baseUinv2R_);
          double cot = digi(ttTrackRef->tanL(), baseUcot_);
          double z0 = digi(ttTrackRef->z0(), baseUzT_);
          // convert stubs
          std::deque<StubTB*> trackStubs;
          for (int layer = 0; layer < numP; layer++) {
            const tt::FrameStub& frameStub = streamsStub[offsetStub + layer][frame];
            const TTStubRef& ttStubRef = frameStub.first;
            if (ttStubRef.isNull())
              continue;
            // parse residuals from tt::Frame and take layerId from tt::TTStubRef
            const bool barrel = setup_->barrel(ttStubRef);
            const int layerIdTracklet = setup_->trackletLayerId(ttStubRef);
            const double baseRZ = barrel ? settings_.kz(layerIdTracklet) : settings_.kz();
            const int widthRZ = barrel ? settings_.zresidbits() : settings_.rresidbits();
            TTBV hw(frameStub.second);
            const TTBV hwRZ(hw, widthRZ, 0, true);
            hw >>= widthRZ;
            hw >>= settings_.phiresidbits();
            const int indexLayerId = setup_->indexLayerId(ttStubRef);
            const tt::SensorModule::Type type = setup_->type(ttStubRef);
            const int widthR = setup_->tbWidthR(type);
            const double baseR = setup_->hybridBaseR(type);
            const TTBV hwR(hw, widthR, 0, barrel);
            double r = hwR.val(baseR) + (barrel ? setup_->hybridLayerR(indexLayerId) : 0.0);
            if (type == tt::SensorModule::Disk2S)
              r = setup_->disk2SR(indexLayerId, r);
            r = digi(r, baseUr_);
            double z = digi(hwRZ.val(baseRZ) * (barrel ? 1. : -cot), baseUz_);
            tbStubs.emplace_back(ttStubRef, r, z, inv2R);
            trackStubs.push_back(&tbStubs.back());
          }
          // create fake seed stubs, since TrackBuilder doesn't output these stubs, required by the KF.
          for (int seedingLayer = 0; seedingLayer < channelAssignment_->numSeedingLayers(); seedingLayer++) {
            const int channelStub = numP + seedingLayer;
            const tt::FrameStub& frameStub = streamsStub[offsetStub + channelStub][frame];
            const TTStubRef& ttStubRef = frameStub.first;
            const int layerId = channelAssignment_->layerId(channel, channelStub);
            const bool barrel = setup_->barrel(ttStubRef);
            double r;
            if (barrel) {
              const int index = layerId - setup_->offsetLayerId();
              const double layer = digi(setup_->hybridLayerR(index), baseUr_);
              const double z = digi(z0 + layer * cot, baseUz_);
              if (std::abs(z) < digi(setup_->tbMinZ(), baseUz_) || index > 0)
                r = digi(setup_->hybridLayerR(index), baseUr_);
              else
                r = digi(setup_->innerRadius(), baseUr_);
            } else {
              const int index = layerId - setup_->offsetLayerId() - setup_->offsetLayerDisks();
              const double side = cot < 0. ? -1. : 1.;
              const double disk = digi(setup_->hybridDiskZ(index), baseUzT_);
              const double invCot = digi(1. / digi(std::abs(cot), baseScot_), baseInvCot_);
              r = digi((disk - side * z0) * invCot, baseUr_);
            }
            tbStubs.emplace_back(ttStubRef, r, 0., inv2R, true);
            trackStubs.push_back(&tbStubs.back());
          }
          // create track
          tbTracks.emplace_back(ttTrackRef, channel, inv2R, cot, z0, trackStubs);
        }
      }
    }
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerTB);
