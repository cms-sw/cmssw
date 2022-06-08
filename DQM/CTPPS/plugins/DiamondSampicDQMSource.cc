/****************************************************************************
 *
 * This is a part of CTPPSDQM software.
 * Authors:
 *   Christopher Misan
 *   Nicola Minafra
 *   Laurent Forthomme
 *
 ****************************************************************************/

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

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingLocalTrack.h"
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

class DiamondSampicDQMSource : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<totemds::Cache>> {
public:
  DiamondSampicDQMSource(const edm::ParameterSet &);
  ~DiamondSampicDQMSource() override;

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
  static const double SAMPIC_ADC_V;

  edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> tokenLocalTrack_;
  edm::EDGetTokenT<edm::DetSetVector<TotemTimingDigi>> tokenDigi_;
  edm::EDGetTokenT<edm::DetSetVector<TotemTimingRecHit>> tokenRecHit_;
  edm::EDGetTokenT<edm::DetSetVector<TotemTimingLocalTrack>> tokenTrack_;
  edm::EDGetTokenT<std::vector<TotemFEDInfo>> tokenFEDInfo_;

  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> ctppsGeometryRunToken_;

  unsigned int samplesForNoise_;
  unsigned int verbosity_;
  bool plotOnline_;
  bool perLSsaving_;  //to avoid nanoDQMIO crashing, driven by  DQMServices/Core/python/DQMStore_cfi.py
  unsigned int trackCorrelationThreshold_;
  edm::TimeValue_t timeOfPreviousEvent_;

  std::unordered_map<unsigned int, double> horizontalShiftOfDiamond_;

  /// plots related to the whole system
  struct GlobalPlots {
    MonitorElement *digiSentPercentage = nullptr;

    GlobalPlots() {}
    GlobalPlots(DQMStore::IBooker &ibooker);
  };

  GlobalPlots globalPlot_;

  struct SectorPlots {
    // Tracks
    MonitorElement *trackCorrelation = nullptr;
    MonitorElement *trackCorrelationLowMultiplicity = nullptr;
    MonitorElement *digiSentPercentage = nullptr;
    SectorPlots(){};
    SectorPlots(DQMStore::IBooker &ibooker, unsigned int id, bool plotOnline);
  };
  std::unordered_map<unsigned int, SectorPlots> sectorPlots_;

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
    MonitorElement *triggerCellTime = nullptr;
    MonitorElement *meanAmplitude = nullptr;
    MonitorElement *cellOfMax = nullptr;

    MonitorElement *hitRate = nullptr;

    MonitorElement *planesWithDigis = nullptr;
    MonitorElement *planesWithTime = nullptr;

    MonitorElement *trackDistribution = nullptr;

    std::set<unsigned int> planesWithDigisSet;
    std::set<unsigned int> planesWithTimeSet;

    PotPlots(){};
    PotPlots(DQMStore::IBooker &ibooker, unsigned int id, bool plotOnline);
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
    MonitorElement *triggerCellTime = nullptr;
    MonitorElement *recHitTime = nullptr;
    MonitorElement *amplitude = nullptr;
    MonitorElement *noiseSamples = nullptr;

    //MonitorElement *hitTime = nullptr;
    MonitorElement *hitRate = nullptr;

    ChannelPlots() {}
    ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, ChannelPlots> channelPlots_;
  static std::string changePathToSampic(std::string path);
};
//----------------------------------------------------------------------------------------------------
std::string DiamondSampicDQMSource::changePathToSampic(std::string path) {
  std::string toReplace = "TimingDiamond";
  path = path.substr(path.find(toReplace) + toReplace.length());
  path = "CTPPS/DiamondSampic/" + path;
  return path;
}
//----------------------------------------------------------------------------------------------------

// Values for all constants
const double DiamondSampicDQMSource::SEC_PER_LUMI_SECTION = 23.31;
const double DiamondSampicDQMSource::LHC_CLOCK_PERIOD_NS = 24.95;
const double DiamondSampicDQMSource::DQM_FRACTION_OF_EVENTS = 1.;
const double DiamondSampicDQMSource::HIT_RATE_FACTOR = DQM_FRACTION_OF_EVENTS / SEC_PER_LUMI_SECTION;
const double DiamondSampicDQMSource::DISPLAY_RESOLUTION_FOR_HITS_MM = 0.05;
const double DiamondSampicDQMSource::INV_DISPLAY_RESOLUTION_FOR_HITS_MM = 1. / DISPLAY_RESOLUTION_FOR_HITS_MM;
const double DiamondSampicDQMSource::SAMPIC_ADC_V = 1. / 256;
//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::GlobalPlots::GlobalPlots(DQMStore::IBooker &ibooker) {
  ibooker.setCurrentFolder("CTPPS/DiamondSampic");
  digiSentPercentage = ibooker.book2D(
      "sent digis percentage", "sent digis percentage (sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::SectorPlots::SectorPlots(DQMStore::IBooker &ibooker, unsigned int id, bool plotOnline) {
  std::string path, title;
  CTPPSDiamondDetId(id).armName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(DiamondSampicDQMSource::changePathToSampic(path));

  CTPPSDiamondDetId(id).armName(title, CTPPSDiamondDetId::nFull);

  trackCorrelation = ibooker.book2D("tracks correlation near-far",
                                    title + " tracks correlation near-far;x (mm);x (mm)",
                                    19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                    -1,
                                    18,
                                    19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                    -1,
                                    18);
  trackCorrelationLowMultiplicity =
      ibooker.book2D("tracks correlation with low multiplicity near-far",
                     title + " tracks correlation with low multiplicity near-far;x (mm);x (mm)",
                     19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                     -1,
                     18,
                     19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                     -1,
                     18);
  if (plotOnline)
    digiSentPercentage = ibooker.book2D("sent digis percentage",
                                        title + " sent digis percentage (sampic);board + 0.5 sampic;channel",
                                        14,
                                        -0.5,
                                        6.5,
                                        16,
                                        0,
                                        16);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id, bool plotOnline) {
  std::string path, title;
  CTPPSDiamondDetId(id).rpName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(DiamondSampicDQMSource::changePathToSampic(path));

  CTPPSDiamondDetId(id).rpName(title, CTPPSDiamondDetId::nFull);

  digiDistribution =
      ibooker.book2D("digi distribution", title + " digi distribution;plane;channel", 10, -0.5, 4.5, 12, 0, 12);

  hitDistribution2d = ibooker.book2D("hits in planes",
                                     title + " hits in planes;plane number;x (mm)",
                                     10,
                                     -0.5,
                                     4.5,
                                     19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                     -0.5,
                                     18.5);
  hitDistribution2dWithTime = ibooker.book2D("hits in planes with time",
                                             title + " hits in planes with time;plane number;x (mm)",
                                             10,
                                             -0.5,
                                             4.5,
                                             19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                             -0.5,
                                             18.5);

  recHitTime = ibooker.book1D("recHit time", title + " recHit time; t (ns)", 500, -25, 25);
  trackDistribution = ibooker.book1D(
      "tracks", title + " tracks;x (mm)", 19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -1, 18);  //TODO needs tracks

  if (plotOnline) {
    hitDistribution2d_lumisection =
        ibooker.book2D("hits in planes lumisection",
                       title + " hits in planes in the last lumisection;plane number;x (mm)",
                       18,
                       -0.5,
                       4,
                       15. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                       0,
                       15);
    triggerCellTime = ibooker.book1D("trigger cell time", title + " Trigger Cell Time; t (ns)", 390, -25, 25);
    activityPerBX = ibooker.book1D("activity per BX CMS", title + " Activity per BX;Event.BX", 3600, -1.5, 3598. + 0.5);
    amplitude = ibooker.book1D("amplitude", title + " amplitude above baseline; amplitude (V)", 50, 0, 1);
    baselineRMS = ibooker.book2D("noise RMS", title + " noise RMS (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
    meanAmplitude =
        ibooker.book2D("mean amplitude", title + " Mean Amplitude (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
    cellOfMax = ibooker.book2D("cell of max", title + " cell of max (0-23);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

    //hitRate = ibooker.book2D("hit rate", title + " hit rate (Hz);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

    planesWithDigis = ibooker.book1D("active planes digis",
                                     title + " active planes with digis sent (per event);number of active planes",
                                     6,
                                     -0.5,
                                     5.5);
    planesWithTime = ibooker.book1D(
        "active planes with time", title + " active planes with time (per event);number of active planes", 6, -0.5, 5.5);

    dataSamplesRaw = ibooker.book1D("raw Samples", title + " Raw Samples; ADC", 256, 0, 256);

    baseline = ibooker.book2D("baseline", title + " baseline (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);
    noiseRMS = ibooker.book2D("noise RMS", title + " noise RMS (V);plane;channel", 10, -0.5, 4.5, 12, 0, 12);

    digiSent = ibooker.book2D(
        "digis sent", title + " digi sent (sampic);board + 0.5 sampic;channel", 14, -0.5, 6.5, 16, 0, 16);
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
  }
}

//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  CTPPSDiamondDetId(id).planeName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(DiamondSampicDQMSource::changePathToSampic(path));

  CTPPSDiamondDetId(id).planeName(title, CTPPSDiamondDetId::nFull);

  digiDistribution = ibooker.book1D("digi distribution", title + " digi distribution;channel", 12, 0, 12);

  hitProfile = ibooker.book1D("hit distribution with time",
                              title + " hit distribution (with time);x(mm)",
                              30. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                              0,
                              30);

  hitMultiplicity = ibooker.book1D("channels per plane", title + " channels per plane; ch per plane", 13, -0.5, 12.5);

  hitMultiplicityWithTime = ibooker.book1D(
      "channels per plane with time", title + " channels per plane with time; ch per plane", 13, -0.5, 12.5);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::ChannelPlots::ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  CTPPSDiamondDetId(id).channelName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(DiamondSampicDQMSource::changePathToSampic(path));

  CTPPSDiamondDetId(id).channelName(title, CTPPSDiamondDetId::nFull);

  activityPerBX = ibooker.book1D("activity per BX", title + " Activity per BX;Event.BX", 1000, -1.5, 998. + 0.5);
  dataSamplesRaw = ibooker.book1D("raw samples", title + " Raw Samples; ADC", 256, 0, 256);
  cellOfMax = ibooker.book1D("cell of max", title + " cell of max; cell", 24, 0, 24);

  triggerCellTime = ibooker.book1D("sampic trigger time", title + " Sampic Trigger Time; t (ns)", 100, -25, 25);
  recHitTime = ibooker.book1D("recHit Time", title + " recHit Time; t (ns)", 500, -25, 25);
  amplitude = ibooker.book1D("amplitude", title + " amplitude above baseline; amplitude (V)", 50, 0, 1);
  noiseSamples = ibooker.book1D("noise samples", title + " noise samples; V", 50, 0, 1);

  //hitTime = ibooker.book1D("hit time", title + "hit time;t - t_previous (us)", 100, 0, 10000);
  //hitRate = ibooker.book1D("hit rate", title + "hit rate;rate (Hz)", 100, 0, 10000);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::DiamondSampicDQMSource(const edm::ParameterSet &ps)
    : tokenLocalTrack_(consumes<edm::DetSetVector<TotemRPLocalTrack>>(ps.getParameter<edm::InputTag>("tagLocalTrack"))),
      tokenDigi_(consumes<edm::DetSetVector<TotemTimingDigi>>(ps.getParameter<edm::InputTag>("tagDigi"))),
      tokenRecHit_(consumes<edm::DetSetVector<TotemTimingRecHit>>(ps.getParameter<edm::InputTag>("tagRecHits"))),
      tokenTrack_(consumes<edm::DetSetVector<TotemTimingLocalTrack>>(ps.getParameter<edm::InputTag>("tagTracks"))),
      tokenFEDInfo_(consumes<std::vector<TotemFEDInfo>>(ps.getParameter<edm::InputTag>("tagFEDInfo"))),
      ctppsGeometryRunToken_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord, edm::Transition::BeginRun>()),
      samplesForNoise_(ps.getUntrackedParameter<unsigned int>("samplesForNoise", 5)),
      verbosity_(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      plotOnline_(ps.getUntrackedParameter<bool>("plotOnline", true)),
      perLSsaving_(ps.getUntrackedParameter<bool>("perLSsaving", false)),
      trackCorrelationThreshold_(ps.getUntrackedParameter<unsigned int>("trackCorrelationThreshold", 3)),
      timeOfPreviousEvent_(0) {}

//----------------------------------------------------------------------------------------------------

DiamondSampicDQMSource::~DiamondSampicDQMSource() {}

//----------------------------------------------------------------------------------------------------

void DiamondSampicDQMSource::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  // Get detector shifts from the geometry (if present)
  const CTPPSGeometry *geom = &iSetup.getData(ctppsGeometryRunToken_);
  for (auto it = geom->beginSensor(); it != geom->endSensor(); it++) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);

    const DetGeomDesc *det = geom->sensorNoThrow(detid);
    if (det)
      horizontalShiftOfDiamond_[detid.rpId()] = det->translation().x() - det->getDiamondDimensions().xHalfWidth;
    else
      edm::LogProblem("DiamondSampicCalibrationDQMSource") << "ERROR: no descriptor for detId";
  }
  //horizontalShiftOfDiamond_=0;//unlock the shift
}

//----------------------------------------------------------------------------------------------------

void DiamondSampicDQMSource::bookHistograms(DQMStore::IBooker &ibooker,
                                            const edm::Run &,
                                            const edm::EventSetup &iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS/DiamondSampic");

  const CTPPSGeometry *geom = &iSetup.getData(ctppsGeometryRunToken_);
  for (auto it = geom->beginSensor(); it != geom->endSensor(); ++it) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);

    sectorPlots_[detid.armId()] = SectorPlots(ibooker, detid.armId(), plotOnline_);

    const CTPPSDiamondDetId rpId(detid.arm(), detid.station(), detid.rp());
    potPlots_[rpId] = PotPlots(ibooker, rpId, plotOnline_);

    if (plotOnline_) {
      globalPlot_ = GlobalPlots(ibooker);
      const CTPPSDiamondDetId plId(detid.arm(), detid.station(), detid.rp(), detid.plane());
      planePlots_[plId] = PlanePlots(ibooker, plId);

      const CTPPSDiamondDetId chId(detid.arm(), detid.station(), detid.rp(), detid.plane(), detid.channel());
      channelPlots_[chId] = ChannelPlots(ibooker, chId);
    }
  }
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<totemds::Cache> DiamondSampicDQMSource::globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                                                   const edm::EventSetup &) const {
  auto d = std::make_shared<totemds::Cache>();
  d->hitDistribution2dMap.reserve(potPlots_.size());
  if (!perLSsaving_ && plotOnline_)
    for (auto &plot : potPlots_)
      d->hitDistribution2dMap[plot.first] =
          std::unique_ptr<TH2F>(static_cast<TH2F *>(plot.second.hitDistribution2d_lumisection->getTH2F()->Clone()));
  return d;
}

//----------------------------------------------------------------------------------------------------

void DiamondSampicDQMSource::analyze(const edm::Event &event, const edm::EventSetup &eventSetup) {
  // get event data
  edm::Handle<edm::DetSetVector<TotemTimingDigi>> timingDigis;
  event.getByToken(tokenDigi_, timingDigis);

  edm::Handle<std::vector<TotemFEDInfo>> fedInfo;
  event.getByToken(tokenFEDInfo_, fedInfo);

  edm::Handle<edm::DetSetVector<TotemTimingRecHit>> timingRecHits;
  event.getByToken(tokenRecHit_, timingRecHits);

  edm::Handle<edm::DetSetVector<TotemTimingLocalTrack>> timingLocalTracks;
  event.getByToken(tokenTrack_, timingLocalTracks);

  // check validity
  bool valid = true;
  valid &= timingDigis.isValid();
  //valid &= fedInfo.isValid();

  if (!valid) {
    if (verbosity_) {
      edm::LogProblem("DiamondSampicDQMSource")
          << "ERROR in DiamondSampicDQMSource::analyze > some of the required inputs "
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
    const CTPPSDiamondDetId detId(digis.detId());
    CTPPSDiamondDetId detId_pot(digis.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    CTPPSDiamondDetId detId_plane(digis.detId());
    detId_plane.setChannel(0);

    for (const auto &digi : digis) {
      // Pot Plots
      if (potPlots_.find(detId_pot) != potPlots_.end()) {
        potPlots_[detId_pot].digiDistribution->Fill(detId.plane(), detId.channel());

        if (plotOnline_) {
          potPlots_[detId_pot].activityPerBX->Fill(event.bunchCrossing());

          for (auto it = digi.samplesBegin(); it != digi.samplesEnd(); ++it) {
            potPlots_[detId_pot].dataSamplesRaw->Fill(*it);
          }

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
      }

      if (plotOnline_) {
        // Plane Plots
        if (planePlots_.find(detId_plane) != planePlots_.end()) {
          planePlots_[detId_plane].digiDistribution->Fill(detId.channel());

          if (channelsPerPlane.find(detId_plane) != channelsPerPlane.end())
            channelsPerPlane[detId_plane]++;
          else
            //if it's the first channel, create new map element with the value of 1
            channelsPerPlane[detId_plane] = 1;
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

          // if (timeOfPreviousEvent_ != 0)
          //   channelPlots_[detId].hitTime->Fill(1e-3 * LHC_CLOCK_PERIOD_NS *
          //                                      (event.time().value() - timeOfPreviousEvent_));
          ++(lumiCache->hitsCounterMap[detId]);
        }
      }
    }
  }
  // End digis

  for (const auto &rechits : *timingRecHits) {
    const CTPPSDiamondDetId detId(rechits.detId());
    CTPPSDiamondDetId detId_pot(rechits.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    CTPPSDiamondDetId detId_plane(rechits.detId());
    detId_plane.setChannel(0);

    for (const auto &rechit : rechits) {
      if (potPlots_.find(detId_pot) != potPlots_.end()) {
        float UFSDShift = 0.0;
        if (rechit.yWidth() < 3)
          UFSDShift = 0.5;

        TH2F *hitHistoTmp = potPlots_[detId_pot].hitDistribution2d->getTH2F();
        TAxis *hitHistoTmpYAxis = hitHistoTmp->GetYaxis();
        int startBin =
            hitHistoTmpYAxis->FindBin(rechit.x() - horizontalShiftOfDiamond_[detId_pot] - 0.5 * rechit.xWidth());
        const int numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i) {
          potPlots_[detId_pot].hitDistribution2d->Fill(detId.plane() + UFSDShift,
                                                       hitHistoTmpYAxis->GetBinCenter(startBin + i));
          if (!perLSsaving_ && plotOnline_)
            potPlots_[detId_pot].hitDistribution2d_lumisection->Fill(detId.plane() + UFSDShift,
                                                                     hitHistoTmpYAxis->GetBinCenter(startBin + i));
        }

        //All plots with Time
        if (rechit.time() != TotemTimingRecHit::NO_T_AVAILABLE) {
          for (int i = 0; i < numOfBins; ++i)
            potPlots_[detId_pot].hitDistribution2dWithTime->Fill(detId.plane() + UFSDShift,
                                                                 hitHistoTmpYAxis->GetBinCenter(startBin + i));

          potPlots_[detId_pot].recHitTime->Fill(rechit.time());
          if (plotOnline_) {
            potPlots_[detId_pot].amplitude->Fill(rechit.amplitude());
            potPlots_[detId_pot].planesWithTimeSet.insert(detId.plane());

            // Plane Plots
            if (planePlots_.find(detId_plane) != planePlots_.end()) {
              TH1F *hitProfileHistoTmp = planePlots_[detId_plane].hitProfile->getTH1F();
              const int startBin = hitProfileHistoTmp->FindBin(rechit.x() - horizontalShiftOfDiamond_[detId_pot] -
                                                               0.5 * rechit.xWidth());
              for (int i = 0; i < numOfBins; ++i)
                hitProfileHistoTmp->Fill(hitProfileHistoTmp->GetBinCenter(startBin + i));

              if (channelsPerPlaneWithTime.find(detId_plane) != channelsPerPlaneWithTime.end())
                channelsPerPlaneWithTime[detId_plane]++;
              else
                //if it's the first channel, create new map element with the value of 1
                channelsPerPlaneWithTime[detId_plane] = 1;
            }

            if (channelPlots_.find(detId) != channelPlots_.end()) {
              potPlots_[detId_pot].triggerCellTime->Fill(rechit.sampicThresholdTime());
              channelPlots_[detId].triggerCellTime->Fill(rechit.sampicThresholdTime());
              channelPlots_[detId].recHitTime->Fill(rechit.time());
              channelPlots_[detId].amplitude->Fill(rechit.amplitude());
            }
          }
        }
      }
    }
  }
  // End RecHits

  // Using CTPPSDiamondLocalTrack
  for (const auto &tracks : *timingLocalTracks) {
    CTPPSDiamondDetId detId_pot(tracks.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    const CTPPSDiamondDetId detId_near(tracks.detId());

    for (const auto &track : tracks) {
      if (!track.isValid())
        continue;
      if (potPlots_.find(detId_pot) == potPlots_.end())
        continue;

      TH1F *trackHistoInTimeTmp = potPlots_[detId_pot].trackDistribution->getTH1F();
      const int startBin =
          trackHistoInTimeTmp->FindBin(track.x0() - horizontalShiftOfDiamond_[detId_pot] - track.x0Sigma());
      const int numOfBins = 2 * track.x0Sigma() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
      for (int i = 0; i < numOfBins; ++i) {
        trackHistoInTimeTmp->Fill(trackHistoInTimeTmp->GetBinCenter(startBin + i));
      }

      //this plot was made with 2 stations per arm in mind
      for (const auto &tracks_far : *timingLocalTracks) {
        CTPPSDiamondDetId detId_far(tracks_far.detId());
        detId_far.setPlane(0);
        detId_far.setChannel(0);
        if (detId_near.arm() != detId_far.arm() || detId_near.station() == detId_far.station())
          continue;
        for (const auto &track_far : tracks_far) {
          if (!track.isValid())
            continue;
          if (sectorPlots_.find(detId_far.armId()) == sectorPlots_.end())
            continue;
          TH2F *trackHistoTmp = sectorPlots_[detId_far.armId()].trackCorrelation->getTH2F();
          TAxis *trackHistoTmpXAxis = trackHistoTmp->GetXaxis();
          TAxis *trackHistoTmpYAxis = trackHistoTmp->GetYaxis();
          const int startBin_far =
              trackHistoTmpYAxis->FindBin(track_far.x0() - horizontalShiftOfDiamond_[detId_far] - track_far.x0Sigma());
          const int numOfBins_far = 2 * track_far.x0Sigma() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
          for (int i = 0; i < numOfBins; ++i) {
            for (int y = 0; y < numOfBins_far; ++y) {
              trackHistoTmp->Fill(trackHistoTmpXAxis->GetBinCenter(startBin + i),
                                  trackHistoTmpYAxis->GetBinCenter(startBin_far + y));
              if (tracks.size() < 3 && tracks_far.size() < trackCorrelationThreshold_)
                sectorPlots_[detId_far.armId()].trackCorrelationLowMultiplicity->Fill(
                    trackHistoTmpXAxis->GetBinCenter(startBin + i), trackHistoTmpYAxis->GetBinCenter(startBin_far + y));
            }
          }
        }
      }
    }
  }
  if (plotOnline_) {
    for (auto &plt : potPlots_) {
      plt.second.planesWithDigis->Fill(plt.second.planesWithDigisSet.size());
      plt.second.planesWithDigisSet.clear();
      plt.second.planesWithTime->Fill(plt.second.planesWithTimeSet.size());
      plt.second.planesWithTimeSet.clear();
    }

    for (const auto &plt : channelsPerPlane)
      planePlots_[plt.first].hitMultiplicity->Fill(plt.second);

    for (const auto &plt : channelsPerPlaneWithTime)
      planePlots_[plt.first].hitMultiplicityWithTime->Fill(plt.second);
  }
  timeOfPreviousEvent_ = event.time().value();
}

//----------------------------------------------------------------------------------------------------

void DiamondSampicDQMSource::globalEndLuminosityBlock(const edm::LuminosityBlock &iLumi, const edm::EventSetup &) {
  auto lumiCache = luminosityBlockCache(iLumi.index());
  if (!perLSsaving_ && plotOnline_) {
    for (auto &plot : potPlots_)
      *(plot.second.hitDistribution2d_lumisection->getTH2F()) = *(lumiCache->hitDistribution2dMap[plot.first]);
    globalPlot_.digiSentPercentage->Reset();
    for (auto &plot : sectorPlots_)
      plot.second.digiSentPercentage->Reset();
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
      //plot.second.hitRate->Reset();
      CTPPSDiamondDetId rpId(plot.first);
      TH2F *hitHistoSectorTmp = sectorPlots_[rpId.armId()].digiSentPercentage->getTH2F();
      hitHistoSectorTmp->Add(hitHistoTmp, 1);

      for (auto &chPlot : channelPlots_) {
        CTPPSDiamondDetId chId(chPlot.first);
        if (chId.arm() == rpId.arm() && chId.rp() == rpId.rp()) {
          plot.second.baseline->Fill(chId.plane(), chId.channel(), chPlot.second.noiseSamples->getTH1F()->GetMean());
          plot.second.noiseRMS->Fill(chId.plane(), chId.channel(), chPlot.second.noiseSamples->getTH1F()->GetRMS());
          plot.second.meanAmplitude->Fill(chId.plane(), chId.channel(), chPlot.second.amplitude->getTH1F()->GetMean());
          plot.second.cellOfMax->Fill(chId.plane(), chId.channel(), chPlot.second.cellOfMax->getTH1F()->GetMean());
          //auto hitsCounterPerLumisection = lumiCache->hitsCounterMap[chPlot.first];
          //plot.second.hitRate->Fill(chId.plane(), chId.channel(), (double)hitsCounterPerLumisection * HIT_RATE_FACTOR);
        }
      }
    }

    // for (auto &plot : channelPlots_) {
    //   auto hitsCounterPerLumisection = lumiCache->hitsCounterMap[plot.first];
    //   if (hitsCounterPerLumisection != 0) {
    //     plot.second.hitRate->Fill((double)hitsCounterPerLumisection * HIT_RATE_FACTOR);
    //   }
    // }
  }
}

DEFINE_FWK_MODULE(DiamondSampicDQMSource);
