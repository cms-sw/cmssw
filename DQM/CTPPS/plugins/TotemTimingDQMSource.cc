/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Nicola Minafra
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Provenance/interface/EventRange.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include <string>

//----------------------------------------------------------------------------------------------------

namespace totemds {
  struct Cache {
    std::unordered_map<unsigned int, std::unique_ptr<TH2F>> hitDistribution2dMap;

    std::unordered_map<unsigned int, unsigned long> hitsCounterMap;
  };
}  // namespace totemds

class TotemTimingDQMSource : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<totemds::Cache>> {
public:
  TotemTimingDQMSource(const edm::ParameterSet &);
  ~TotemTimingDQMSource() override;

protected:
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  std::shared_ptr<totemds::Cache> globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                             const edm::EventSetup &) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;

private:
  // Constants
  static const double SEC_PER_LUMI_SECTION;  // Number of seconds per
                                             // lumisection: used to compute hit
                                             // rates in Hz
  static const double LHC_CLOCK_PERIOD_NS;
  static const double DQM_FRACTION_OF_EVENTS;          // approximate fraction of events
                                                       // sent to DQM stream
  static const double HIT_RATE_FACTOR;                 // factor to have real rate in Hz
  static const double DISPLAY_RESOLUTION_FOR_HITS_MM;  // Bin width of histograms
                                                       // showing hits and tracks
                                                       // (in mm)
  static const double INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
  static const double TOMOGRAPHY_RESOLUTION_MM;
  static const double SAMPIC_SAMPLING_PERIOD_NS;  // ns per HPTDC bin
  static const double SAMPIC_MAX_NUMBER_OF_SAMPLES;
  static const double SAMPIC_ADC_V;
  static const int CTPPS_NUM_OF_ARMS;
  static const int TOTEM_TIMING_STATION_ID;
  static const int TOTEM_STATION_210;
  static const int TOTEM_STATION_220;
  static const int TOTEM_TIMING_TOP_RP_ID;
  static const int TOTEM_TIMING_BOT_RP_ID;
  static const int TOTEM_STRIP_MIN_RP_ID;
  static const int TOTEM_STRIP_MAX_RP_ID;
  static const int CTPPS_NEAR_RP_ID;
  static const int CTPPS_FAR_RP_ID;
  static const int TOTEM_TIMING_NUM_OF_PLANES;
  static const int TOTEM_TIMING_NUM_OF_CHANNELS;
  static const int TOTEM_TIMING_FED_ID_45;
  static const int TOTEM_TIMING_FED_ID_56;
  static const float COS_8_DEG;
  static const float SIN_8_DEG;

  edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> tokenLocalTrack_;
  edm::EDGetTokenT<edm::DetSetVector<TotemTimingDigi>> tokenDigi_;
  edm::EDGetTokenT<edm::DetSetVector<TotemTimingRecHit>> tokenRecHit_;
  edm::EDGetTokenT<std::vector<TotemFEDInfo>> tokenFEDInfo_;

  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geometryToken_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geometryTokenBeginRun_;

  double minimumStripAngleForTomography_;
  double maximumStripAngleForTomography_;
  unsigned int samplesForNoise_;
  bool perLSsaving_;  //to avoid nanoDQMIO crashing, driven by  DQMServices/Core/python/DQMStore_cfi.py
  unsigned int verbosity_;
  edm::TimeValue_t timeOfPreviousEvent_;

  float verticalShiftBot_, verticalShiftTop_;

  /// plots related to the whole system
  struct GlobalPlots {
    MonitorElement *digiSentPercentage = nullptr;

    GlobalPlots() {}
    GlobalPlots(DQMStore::IBooker &ibooker);
  };

  GlobalPlots globalPlot_;

  /// plots related to one Diamond detector package
  struct PotPlots {
    // Digis
    MonitorElement *activityPerBX = nullptr;
    MonitorElement *digiDistribution = nullptr;
    MonitorElement *dataSamplesRaw = nullptr;
    MonitorElement *baseline = nullptr;
    MonitorElement *noiseRMS = nullptr;

    MonitorElement *digiSent = nullptr;
    MonitorElement *digiAll = nullptr;
    MonitorElement *digiSentPercentage = nullptr;

    // RecHits
    MonitorElement *hitDistribution2d = nullptr;
    MonitorElement *hitDistribution2dWithTime = nullptr;
    MonitorElement *hitDistribution2d_lumisection = nullptr;

    MonitorElement *recHitTime = nullptr;
    MonitorElement *amplitude = nullptr;
    MonitorElement *baselineRMS = nullptr;
    MonitorElement *tirggerCellTime = nullptr;
    MonitorElement *meanAmplitude = nullptr;
    MonitorElement *cellOfMax = nullptr;

    MonitorElement *hitRate = nullptr;

    MonitorElement *planesWithDigis = nullptr;
    MonitorElement *planesWithTime = nullptr;

    // MonitorElement *trackDistribution = nullptr;

    MonitorElement *stripTomography210 = nullptr;
    MonitorElement *stripTomography220 = nullptr;

    std::set<unsigned int> planesWithDigisSet;
    std::set<unsigned int> planesWithTimeSet;

    PotPlots(){};
    PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, PotPlots> potPlots_;

  /// plots related to one Diamond plane
  struct PlanePlots {
    MonitorElement *digiDistribution = nullptr;

    MonitorElement *hitProfile = nullptr;
    MonitorElement *hitMultiplicity = nullptr;
    MonitorElement *hitMultiplicityWithTime = nullptr;

    PlanePlots() {}
    PlanePlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, PlanePlots> planePlots_;

  /// plots related to one Diamond channel
  struct ChannelPlots {
    // Digis
    MonitorElement *activityPerBX = nullptr;
    MonitorElement *dataSamplesRaw = nullptr;
    MonitorElement *cellOfMax = nullptr;
    MonitorElement *maxTimeAfterTrigger = nullptr;

    // RecHits
    MonitorElement *tirggerCellTime = nullptr;
    MonitorElement *recHitTime = nullptr;
    MonitorElement *amplitude = nullptr;
    MonitorElement *noiseSamples = nullptr;

    MonitorElement *hitTime = nullptr;
    MonitorElement *hitRate = nullptr;

    MonitorElement *stripTomography210 = nullptr;
    MonitorElement *stripTomography220 = nullptr;

    ChannelPlots() {}
    ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, ChannelPlots> channelPlots_;
};

//----------------------------------------------------------------------------------------------------

// Values for all constants
const double TotemTimingDQMSource::SEC_PER_LUMI_SECTION = 23.31;
const double TotemTimingDQMSource::LHC_CLOCK_PERIOD_NS = 24.95;
const double TotemTimingDQMSource::DQM_FRACTION_OF_EVENTS = 1.;
const double TotemTimingDQMSource::HIT_RATE_FACTOR = DQM_FRACTION_OF_EVENTS / SEC_PER_LUMI_SECTION;
const double TotemTimingDQMSource::DISPLAY_RESOLUTION_FOR_HITS_MM = 0.1;
const double TotemTimingDQMSource::INV_DISPLAY_RESOLUTION_FOR_HITS_MM = 1. / DISPLAY_RESOLUTION_FOR_HITS_MM;
const double TotemTimingDQMSource::TOMOGRAPHY_RESOLUTION_MM = 1;
const double TotemTimingDQMSource::SAMPIC_SAMPLING_PERIOD_NS = 1. / 7.8e9;
const double TotemTimingDQMSource::SAMPIC_MAX_NUMBER_OF_SAMPLES = 64;
const double TotemTimingDQMSource::SAMPIC_ADC_V = 1. / 256;
const int TotemTimingDQMSource::CTPPS_NUM_OF_ARMS = 2;
const int TotemTimingDQMSource::TOTEM_TIMING_STATION_ID = 2;
const int TotemTimingDQMSource::TOTEM_STATION_210 = 0;
const int TotemTimingDQMSource::TOTEM_STATION_220 = 2;
const int TotemTimingDQMSource::TOTEM_TIMING_TOP_RP_ID = 0;
const int TotemTimingDQMSource::TOTEM_TIMING_BOT_RP_ID = 1;
const int TotemTimingDQMSource::TOTEM_STRIP_MIN_RP_ID = 4;
const int TotemTimingDQMSource::TOTEM_STRIP_MAX_RP_ID = 5;
const int TotemTimingDQMSource::CTPPS_NEAR_RP_ID = 2;
const int TotemTimingDQMSource::CTPPS_FAR_RP_ID = 3;
const int TotemTimingDQMSource::TOTEM_TIMING_NUM_OF_PLANES = 4;
const int TotemTimingDQMSource::TOTEM_TIMING_NUM_OF_CHANNELS = 12;
const int TotemTimingDQMSource::TOTEM_TIMING_FED_ID_45 = FEDNumbering::MAXTotemRPTimingVerticalFEDID;
const int TotemTimingDQMSource::TOTEM_TIMING_FED_ID_56 = FEDNumbering::MINTotemRPTimingVerticalFEDID;
const float TotemTimingDQMSource::COS_8_DEG = 0.990268;
const float TotemTimingDQMSource::SIN_8_DEG = -0.139173;

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::GlobalPlots::GlobalPlots(DQMStore::IBooker &ibooker) {
  ibooker.setCurrentFolder("CTPPS/TimingFastSilicon");

  digiSentPercentage = ibooker.book2D(
      "sent digis percentage", "sent digis percentage (sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  TotemTimingDetId(id).rpName(path, TotemTimingDetId::nPath);
  ibooker.setCurrentFolder(path);

  TotemTimingDetId(id).rpName(title, TotemTimingDetId::nFull);

  activityPerBX = ibooker.book1D("activity per BX CMS", title + " Activity per BX;Event.BX", 3600, -1.5, 3598. + 0.5);

  digiDistribution =
      ibooker.book2D("digi distribution", title + " digi distribution;plane;channel", 10, -0.5, 4.5, 12, 0, 12);

  dataSamplesRaw = ibooker.book1D("raw Samples", title + " Raw Samples; ADC", 256, 0, 256);

  baseline = ibooker.book2D("baseline", title + " baseline (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  noiseRMS = ibooker.book2D("noise RMS", title + " noise RMS (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

  digiSent =
      ibooker.book2D("digis sent", title + " digi sent (sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
  digiAll =
      ibooker.book2D("all digis", title + " all digis(sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
  digiSentPercentage = ibooker.book2D("sent digis percentage",
                                      title + " sent digis percentage (sampic);board + 0.5 sampic;channel",
                                      14,
                                      -0.5,
                                      6.5,
                                      16,
                                      0,
                                      16);

  hitDistribution2d = ibooker.book2D("hits in planes",
                                     title + " hits in planes;plane number;x (mm)",
                                     18,
                                     -0.5,
                                     4,
                                     15. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                     0,
                                     15);
  hitDistribution2dWithTime = ibooker.book2D("hits in planes with time",
                                             title + " hits in planes with time;plane number;x (mm)",
                                             18,
                                             -0.5,
                                             4,
                                             15. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                             0,
                                             15);
  hitDistribution2d_lumisection = ibooker.book2D("hits in planes lumisection",
                                                 title + " hits in planes in the last lumisection;plane number;x (mm)",
                                                 18,
                                                 -0.5,
                                                 4,
                                                 15. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                                 0,
                                                 15);

  recHitTime = ibooker.book1D("recHit time", title + " time in the recHits; t (ns)", 500, -25, 25);
  amplitude = ibooker.book1D("amplitude", title + " amplitude above baseline; amplitude (V)", 50, 0, 1);
  tirggerCellTime = ibooker.book1D("trigger cell time", title + " Trigger Cell Time; t (ns)", 390, -25, 25);
  baselineRMS = ibooker.book2D("noise RMS", title + " noise RMS (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  meanAmplitude =
      ibooker.book2D("mean amplitude", title + " Mean Amplitude (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
  cellOfMax = ibooker.book2D("cell of max", title + " cell of max (0-23);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

  hitRate = ibooker.book2D("hit rate", title + " hit rate (Hz);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

  planesWithDigis = ibooker.book1D(
      "active planes digis", title + " active planes with digis sent (per event);number of active planes", 6, -0.5, 5.5);
  planesWithTime = ibooker.book1D(
      "active planes with time", title + " active planes with time (per event);number of active planes", 6, -0.5, 5.5);

  // trackDistribution = ibooker.book1D( "tracks", title+" tracks;x (mm)",
  //     19.*INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -1, 18 );    //TODO needs tracks

  stripTomography210 =
      ibooker.book2D("tomography 210",
                     title + " tomography (only with time) with strips 210 (all planes);x + 50*plane(mm);y (mm)",
                     190 / TOMOGRAPHY_RESOLUTION_MM,
                     -20,
                     170,
                     25 / TOMOGRAPHY_RESOLUTION_MM,
                     0,
                     25);
  stripTomography220 =
      ibooker.book2D("tomography 220",
                     title + " tomography (only with time) with strips 220 (all planes);x + 50*plane(mm);y (mm)",
                     190 / TOMOGRAPHY_RESOLUTION_MM,
                     -20,
                     170,
                     25 / TOMOGRAPHY_RESOLUTION_MM,
                     0,
                     25);
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  TotemTimingDetId(id).planeName(path, TotemTimingDetId::nPath);
  ibooker.setCurrentFolder(path);

  TotemTimingDetId(id).planeName(title, TotemTimingDetId::nFull);

  digiDistribution = ibooker.book1D("digi distribution", title + " digi distribution;channel", 12, 0, 12);

  hitProfile = ibooker.book1D("hit distribution with time",
                              title + " hit distribution (with time);y (+ 15 for x>3) (mm)",
                              30. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                              0,
                              30);

  hitMultiplicity = ibooker.book1D("channels per plane", title + " channels per plane; ch per plane", 13, -0.5, 12.5);

  hitMultiplicityWithTime = ibooker.book1D(
      "channels per plane with time", title + " channels per plane with time; ch per plane", 13, -0.5, 12.5);
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::ChannelPlots::ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  TotemTimingDetId(id).channelName(path, TotemTimingDetId::nPath);
  ibooker.setCurrentFolder(path);

  TotemTimingDetId(id).channelName(title, TotemTimingDetId::nFull);

  activityPerBX = ibooker.book1D("activity per BX", title + " Activity per BX;Event.BX", 1000, -1.5, 998. + 0.5);
  dataSamplesRaw = ibooker.book1D("raw samples", title + " Raw Samples; ADC", 256, 0, 256);
  cellOfMax = ibooker.book1D("cell of max", title + " cell of max; cell", 24, 0, 24);

  tirggerCellTime = ibooker.book1D("sampic trigger time", title + " Sampic Trigger Time; t (ns)", 100, -25, 25);
  recHitTime = ibooker.book1D("recHit Time", title + " recHit Time; t (ns)", 500, -25, 25);
  amplitude = ibooker.book1D("amplitude", title + " amplitude above baseline; amplitude (V)", 50, 0, 1);
  noiseSamples = ibooker.book1D("noise samples", title + " noise samples; V", 50, 0, 1);

  hitTime = ibooker.book1D("hit time", title + "hit time;t - t_previous (us)", 100, 0, 10000);
  hitRate = ibooker.book1D("hit rate", title + "hit rate;rate (Hz)", 100, 0, 10000);

  stripTomography210 = ibooker.book2D("tomography 210",
                                      title + " tomography with strips 210;x (mm);y (mm)",
                                      20 / TOMOGRAPHY_RESOLUTION_MM,
                                      -20,
                                      20,
                                      25 / TOMOGRAPHY_RESOLUTION_MM,
                                      0,
                                      25);
  stripTomography220 = ibooker.book2D("tomography 220",
                                      title + " tomography with strips 220;x (mm);y (mm)",
                                      20 / TOMOGRAPHY_RESOLUTION_MM,
                                      -20,
                                      20,
                                      25 / TOMOGRAPHY_RESOLUTION_MM,
                                      0,
                                      25);
}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::TotemTimingDQMSource(const edm::ParameterSet &ps)
    : tokenLocalTrack_(
          consumes<edm::DetSetVector<TotemRPLocalTrack>>(ps.getUntrackedParameter<edm::InputTag>("tagLocalTrack"))),
      tokenDigi_(consumes<edm::DetSetVector<TotemTimingDigi>>(ps.getUntrackedParameter<edm::InputTag>("tagDigi"))),
      tokenRecHit_(
          consumes<edm::DetSetVector<TotemTimingRecHit>>(ps.getUntrackedParameter<edm::InputTag>("tagRecHits"))),
      // tokenTrack_(consumes<edm::DetSetVector<TotemTimingLocalTrack>>(
      //     ps.getParameter<edm::InputTag>("tagLocalTracks"))),
      tokenFEDInfo_(consumes<std::vector<TotemFEDInfo>>(ps.getUntrackedParameter<edm::InputTag>("tagFEDInfo"))),
      geometryToken_(esConsumes()),
      geometryTokenBeginRun_(esConsumes<edm::Transition::BeginRun>()),
      minimumStripAngleForTomography_(ps.getParameter<double>("minimumStripAngleForTomography")),
      maximumStripAngleForTomography_(ps.getParameter<double>("maximumStripAngleForTomography")),
      samplesForNoise_(ps.getUntrackedParameter<unsigned int>("samplesForNoise", 5)),
      perLSsaving_(ps.getUntrackedParameter<bool>("perLSsaving", false)),
      verbosity_(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      timeOfPreviousEvent_(0) {}

//----------------------------------------------------------------------------------------------------

TotemTimingDQMSource::~TotemTimingDQMSource() {}

//----------------------------------------------------------------------------------------------------

void TotemTimingDQMSource::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  // Get detector shifts from the geometry (if present)
  auto const &geom = iSetup.getData(geometryTokenBeginRun_);

  const TotemTimingDetId detid_top(0, TOTEM_TIMING_STATION_ID, TOTEM_TIMING_BOT_RP_ID, 0, 0);
  const TotemTimingDetId detid_bot(0, TOTEM_TIMING_STATION_ID, TOTEM_TIMING_TOP_RP_ID, 0, 7);
  verticalShiftTop_ = 0;
  verticalShiftBot_ = 0;
  {
    const DetGeomDesc *det_top = geom.sensorNoThrow(detid_top);
    if (det_top) {
      verticalShiftTop_ = det_top->translation().y() + det_top->getDiamondDimensions().yHalfWidth;
    }
    const DetGeomDesc *det_bot = geom.sensorNoThrow(detid_bot);
    if (det_bot)
      verticalShiftBot_ = det_bot->translation().y() + det_bot->getDiamondDimensions().yHalfWidth;
  }
}

//----------------------------------------------------------------------------------------------------

void TotemTimingDQMSource::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) {
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  globalPlot_ = GlobalPlots(ibooker);

  for (unsigned short arm = 0; arm < CTPPS_NUM_OF_ARMS; ++arm) {
    for (unsigned short rp = TOTEM_TIMING_TOP_RP_ID; rp <= TOTEM_TIMING_BOT_RP_ID; ++rp) {
      const TotemTimingDetId rpId(arm, TOTEM_TIMING_STATION_ID, rp);
      potPlots_[rpId] = PotPlots(ibooker, rpId);
      for (unsigned short pl = 0; pl < TOTEM_TIMING_NUM_OF_PLANES; ++pl) {
        const TotemTimingDetId plId(arm, TOTEM_TIMING_STATION_ID, rp, pl);
        planePlots_[plId] = PlanePlots(ibooker, plId);
        for (unsigned short ch = 0; ch < TOTEM_TIMING_NUM_OF_CHANNELS; ++ch) {
          const TotemTimingDetId chId(arm, TOTEM_TIMING_STATION_ID, rp, pl, ch);
          channelPlots_[chId] = ChannelPlots(ibooker, chId);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<totemds::Cache> TotemTimingDQMSource::globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                                                 const edm::EventSetup &) const {
  auto d = std::make_shared<totemds::Cache>();
  d->hitDistribution2dMap.reserve(potPlots_.size());
  if (!perLSsaving_) {
    for (auto &plot : potPlots_)
      d->hitDistribution2dMap[plot.first] =
          std::unique_ptr<TH2F>(static_cast<TH2F *>(plot.second.hitDistribution2d_lumisection->getTH2F()->Clone()));
  }
  return d;
}

//----------------------------------------------------------------------------------------------------

void TotemTimingDQMSource::analyze(const edm::Event &event, const edm::EventSetup &eventSetup) {
  // get event setup data
  auto const &geometry = eventSetup.getData(geometryToken_);

  // get event data
  edm::Handle<edm::DetSetVector<TotemRPLocalTrack>> stripTracks;
  event.getByToken(tokenLocalTrack_, stripTracks);

  edm::Handle<edm::DetSetVector<TotemTimingDigi>> timingDigis;
  event.getByToken(tokenDigi_, timingDigis);

  edm::Handle<std::vector<TotemFEDInfo>> fedInfo;
  event.getByToken(tokenFEDInfo_, fedInfo);

  edm::Handle<edm::DetSetVector<TotemTimingRecHit>> timingRecHits;
  event.getByToken(tokenRecHit_, timingRecHits);

  // check validity
  bool valid = true;
  valid &= timingDigis.isValid();
  valid &= fedInfo.isValid();

  if (!valid) {
    if (verbosity_) {
      edm::LogProblem("TotemTimingDQMSource") << "ERROR in TotemTimingDQMSource::analyze > some of the required inputs "
                                                 "are not valid. Skipping this event.\n"
                                              << "    timingDigis.isValid = " << timingDigis.isValid() << "\n"
                                              << "    fedInfo.isValid = " << fedInfo.isValid();
    }

    return;
  }

  // Using TotemTimingDigi
  std::set<uint8_t> boardSet;
  std::unordered_map<unsigned int, unsigned int> channelsPerPlane;
  std::unordered_map<unsigned int, unsigned int> channelsPerPlaneWithTime;

  auto lumiCache = luminosityBlockCache(event.getLuminosityBlock().index());
  for (const auto &digis : *timingDigis) {
    const TotemTimingDetId detId(digis.detId());
    TotemTimingDetId detId_pot(digis.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    TotemTimingDetId detId_plane(digis.detId());
    detId_plane.setChannel(0);

    for (const auto &digi : digis) {
      // Pot Plots
      if (potPlots_.find(detId_pot) != potPlots_.end()) {
        potPlots_[detId_pot].activityPerBX->Fill(event.bunchCrossing());

        potPlots_[detId_pot].digiDistribution->Fill(detId.plane(), detId.channel());

        for (auto it = digi.samplesBegin(); it != digi.samplesEnd(); ++it)
          potPlots_[detId_pot].dataSamplesRaw->Fill(*it);

        float boardId = digi.eventInfo().hardwareBoardId() + 0.5 * digi.eventInfo().hardwareSampicId();
        potPlots_[detId_pot].digiSent->Fill(boardId, digi.hardwareChannelId());
        if (boardSet.find(digi.eventInfo().hardwareId()) == boardSet.end()) {
          // This guarantees that every board is counted only once
          boardSet.insert(digi.eventInfo().hardwareId());
          std::bitset<16> chMap(digi.eventInfo().channelMap());
          for (int i = 0; i < 16; ++i) {
            if (chMap.test(i)) {
              potPlots_[detId_pot].digiAll->Fill(boardId, i);
            }
          }
        }

        potPlots_[detId_pot].planesWithDigisSet.insert(detId.plane());
      }

      // Plane Plots
      if (planePlots_.find(detId_plane) != planePlots_.end()) {
        planePlots_[detId_plane].digiDistribution->Fill(detId.channel());

        if (channelsPerPlane.find(detId_plane) != channelsPerPlane.end())
          channelsPerPlane[detId_plane]++;
        else
          channelsPerPlane[detId_plane] = 0;
      }

      // Channel Plots
      if (channelPlots_.find(detId) != channelPlots_.end()) {
        channelPlots_[detId].activityPerBX->Fill(event.bunchCrossing());

        for (auto it = digi.samplesBegin(); it != digi.samplesEnd(); ++it)
          channelPlots_[detId].dataSamplesRaw->Fill(*it);
        for (unsigned short i = 0; i < samplesForNoise_; ++i)
          channelPlots_[detId].noiseSamples->Fill(SAMPIC_ADC_V * digi.sampleAt(i));

        unsigned int cellOfMax = std::max_element(digi.samplesBegin(), digi.samplesEnd()) - digi.samplesBegin();
        channelPlots_[detId].cellOfMax->Fill((int)cellOfMax);

        if (timeOfPreviousEvent_ != 0)
          channelPlots_[detId].hitTime->Fill(1e-3 * LHC_CLOCK_PERIOD_NS *
                                             (event.time().value() - timeOfPreviousEvent_));
        ++(lumiCache->hitsCounterMap[detId]);
      }
    }
  }
  // End digis

  for (const auto &rechits : *timingRecHits) {
    const TotemTimingDetId detId(rechits.detId());
    TotemTimingDetId detId_pot(rechits.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    TotemTimingDetId detId_plane(rechits.detId());
    detId_plane.setChannel(0);

    for (const auto &rechit : rechits) {
      if (potPlots_.find(detId_pot) != potPlots_.end()) {
        potPlots_[detId_pot].amplitude->Fill(rechit.amplitude());

        TH2F *hitHistoTmp = potPlots_[detId_pot].hitDistribution2d->getTH2F();
        TAxis *hitHistoTmpYAxis = hitHistoTmp->GetYaxis();
        float yCorrected = rechit.y();
        yCorrected += (detId.rp() == TOTEM_TIMING_TOP_RP_ID) ? verticalShiftTop_ : verticalShiftBot_;
        float x_shift = detId.plane();
        x_shift += (rechit.x() > 2) ? 0.25 : 0;
        int startBin = hitHistoTmpYAxis->FindBin(yCorrected - 0.5 * rechit.yWidth());
        int numOfBins = rechit.yWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i) {
          potPlots_[detId_pot].hitDistribution2d->Fill(detId.plane() + 0.25 * (rechit.x() > 2),
                                                       hitHistoTmpYAxis->GetBinCenter(startBin + i));
          if (!perLSsaving_)
            potPlots_[detId_pot].hitDistribution2d_lumisection->Fill(x_shift,
                                                                     hitHistoTmpYAxis->GetBinCenter(startBin + i));
        }

        //All plots with Time
        if (rechit.time() != TotemTimingRecHit::NO_T_AVAILABLE) {
          for (int i = 0; i < numOfBins; ++i)
            potPlots_[detId_pot].hitDistribution2dWithTime->Fill(detId.plane() + 0.25 * (rechit.x() > 2),
                                                                 hitHistoTmpYAxis->GetBinCenter(startBin + i));

          potPlots_[detId_pot].recHitTime->Fill(rechit.time());
          potPlots_[detId_pot].planesWithTimeSet.insert(detId.plane());

          // Plane Plots
          if (planePlots_.find(detId_plane) != planePlots_.end()) {
            // Visualization tricks
            float x_shift = (rechit.x() > 2) ? 15 : 0;
            TH1F *hitProfileHistoTmp = planePlots_[detId_plane].hitProfile->getTH1F();
            int numOfBins = rechit.yWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
            if (detId.rp() == TOTEM_TIMING_TOP_RP_ID) {
              float yCorrected = rechit.y() + verticalShiftTop_ - 0.5 * rechit.yWidth() + x_shift;
              int startBin = hitProfileHistoTmp->FindBin(yCorrected);
              for (int i = 0; i < numOfBins; ++i)
                hitProfileHistoTmp->Fill(hitProfileHistoTmp->GetBinCenter(startBin + i));
            } else {
              float yCorrected = rechit.y() + verticalShiftBot_ + 0.5 * rechit.yWidth() + (15 - x_shift);
              int startBin = hitProfileHistoTmp->FindBin(yCorrected);
              int totBins = hitProfileHistoTmp->GetNbinsX();
              for (int i = 0; i < numOfBins; ++i)
                hitProfileHistoTmp->Fill(hitProfileHistoTmp->GetBinCenter(totBins - startBin + i));
            }

            if (channelsPerPlaneWithTime.find(detId_plane) != channelsPerPlaneWithTime.end())
              channelsPerPlaneWithTime[detId_plane]++;
            else
              channelsPerPlaneWithTime[detId_plane] = 0;
          }

          if (channelPlots_.find(detId) != channelPlots_.end()) {
            potPlots_[detId_pot].tirggerCellTime->Fill(rechit.sampicThresholdTime());
            channelPlots_[detId].tirggerCellTime->Fill(rechit.sampicThresholdTime());
            channelPlots_[detId].recHitTime->Fill(rechit.time());
            channelPlots_[detId].amplitude->Fill(rechit.amplitude());
          }
        }
      }
    }
  }
  // End RecHits

  // Tomography of timing using strips
  for (const auto &rechits : *timingRecHits) {
    const TotemTimingDetId detId(rechits.detId());
    TotemTimingDetId detId_pot(rechits.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    TotemTimingDetId detId_plane(rechits.detId());
    detId_plane.setChannel(0);

    float y_shift = (detId.rp() == TOTEM_TIMING_TOP_RP_ID) ? 20 : 5;

    for (const auto &rechit : rechits) {
      if (rechit.time() != TotemTimingRecHit::NO_T_AVAILABLE && potPlots_.find(detId_pot) != potPlots_.end() &&
          planePlots_.find(detId_plane) != planePlots_.end() && channelPlots_.find(detId) != channelPlots_.end()) {
        if (stripTracks.isValid()) {
          for (const auto &ds : *stripTracks) {
            const CTPPSDetId stripId(ds.detId());
            // mean position of U and V planes
            TotemRPDetId plId_V(stripId);
            plId_V.setPlane(0);
            TotemRPDetId plId_U(stripId);
            plId_U.setPlane(1);

            double rp_x = 0;
            double rp_y = 0;
            try {
              rp_x = (geometry.sensor(plId_V)->translation().x() + geometry.sensor(plId_U)->translation().x()) / 2;
              rp_y = (geometry.sensor(plId_V)->translation().y() + geometry.sensor(plId_U)->translation().y()) / 2;
            } catch (const cms::Exception &) {
              continue;
            }

            for (const auto &striplt : ds) {
              if (striplt.isValid() && stripId.arm() == detId.arm()) {
                if (striplt.tx() > maximumStripAngleForTomography_ || striplt.ty() > maximumStripAngleForTomography_)
                  continue;
                if (striplt.tx() < minimumStripAngleForTomography_ || striplt.ty() < minimumStripAngleForTomography_)
                  continue;
                if (stripId.rp() - detId.rp() == (TOTEM_STRIP_MAX_RP_ID - TOTEM_TIMING_BOT_RP_ID)) {
                  double x = striplt.x0() - rp_x;
                  double y = striplt.y0() - rp_y;
                  if (stripId.station() == TOTEM_STATION_210) {
                    potPlots_[detId_pot].stripTomography210->Fill(x + detId.plane() * 50, y + y_shift);
                    channelPlots_[detId].stripTomography210->Fill(x, y + y_shift);
                  } else if (stripId.station() == TOTEM_STATION_220) {
                    potPlots_[detId_pot].stripTomography220->Fill(x + detId.plane() * 50, y + y_shift);
                    channelPlots_[detId].stripTomography220->Fill(x, y + y_shift);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  for (auto &plt : potPlots_) {
    plt.second.planesWithDigis->Fill(plt.second.planesWithDigisSet.size());
    plt.second.planesWithDigisSet.clear();
    plt.second.planesWithTime->Fill(plt.second.planesWithTimeSet.size());
    plt.second.planesWithTimeSet.clear();
  }

  for (const auto &plt : channelsPerPlane) {
    planePlots_[plt.first].hitMultiplicity->Fill(plt.second);
  }
  for (const auto &plt : channelsPerPlaneWithTime) {
    planePlots_[plt.first].hitMultiplicityWithTime->Fill(plt.second);
  }

  timeOfPreviousEvent_ = event.time().value();
}

//----------------------------------------------------------------------------------------------------

void TotemTimingDQMSource::globalEndLuminosityBlock(const edm::LuminosityBlock &iLumi, const edm::EventSetup &) {
  auto lumiCache = luminosityBlockCache(iLumi.index());
  if (!perLSsaving_) {
    for (auto &plot : potPlots_) {
      *(plot.second.hitDistribution2d_lumisection->getTH2F()) = *(lumiCache->hitDistribution2dMap[plot.first]);
    }

    globalPlot_.digiSentPercentage->Reset();
    TH2F *hitHistoGlobalTmp = globalPlot_.digiSentPercentage->getTH2F();
    for (auto &plot : potPlots_) {
      TH2F *hitHistoTmp = plot.second.digiSentPercentage->getTH2F();
      TH2F *histoSent = plot.second.digiSent->getTH2F();
      TH2F *histoAll = plot.second.digiAll->getTH2F();

      hitHistoTmp->Divide(histoSent, histoAll);
      hitHistoTmp->Scale(100);
      hitHistoGlobalTmp->Add(hitHistoTmp, 1);

      plot.second.baseline->Reset();
      plot.second.noiseRMS->Reset();
      plot.second.meanAmplitude->Reset();
      plot.second.cellOfMax->Reset();
      plot.second.hitRate->Reset();
      TotemTimingDetId rpId(plot.first);
      for (auto &chPlot : channelPlots_) {
        TotemTimingDetId chId(chPlot.first);
        if (chId.arm() == rpId.arm() && chId.rp() == rpId.rp()) {
          plot.second.baseline->Fill(chId.plane(), chId.channel(), chPlot.second.noiseSamples->getTH1F()->GetMean());
          plot.second.noiseRMS->Fill(chId.plane(), chId.channel(), chPlot.second.noiseSamples->getTH1F()->GetRMS());
          plot.second.meanAmplitude->Fill(chId.plane(), chId.channel(), chPlot.second.amplitude->getTH1F()->GetMean());
          plot.second.cellOfMax->Fill(chId.plane(), chId.channel(), chPlot.second.cellOfMax->getTH1F()->GetMean());
          auto hitsCounterPerLumisection = lumiCache->hitsCounterMap[chPlot.first];
          plot.second.hitRate->Fill(chId.plane(), chId.channel(), (double)hitsCounterPerLumisection * HIT_RATE_FACTOR);
        }
      }
    }

    for (auto &plot : channelPlots_) {
      auto hitsCounterPerLumisection = lumiCache->hitsCounterMap[plot.first];
      if (hitsCounterPerLumisection != 0) {
        plot.second.hitRate->Fill((double)hitsCounterPerLumisection * HIT_RATE_FACTOR);
      }
    }
  }
}

DEFINE_FWK_MODULE(TotemTimingDQMSource);
