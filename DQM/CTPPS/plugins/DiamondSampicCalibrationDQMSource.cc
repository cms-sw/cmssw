// -*- C++ -*-
//
// Package:    DQM/CTPPS/DiamondSampicCalibrationDQMSource
// Class:      DiamondSampicCalibrationDQMSource
//
/**\class DiamondSampicCalibrationDQMSource DiamondSampicCalibrationDQMSource.cc SampicDigi/DiamondSampicCalibrationDQMSource/plugins/DiamondSampicCalibrationDQMSource.cc

 Description: DQM module for the diamond sampic offset calibration

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Misan
//         Created:  Mon, 24 Aug 2021 14:21:17 GMT
//
//

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

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingLocalTrack.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class DiamondSampicCalibrationDQMSource : public DQMOneEDAnalyzer<> {
public:
  DiamondSampicCalibrationDQMSource(const edm::ParameterSet &);
  ~DiamondSampicCalibrationDQMSource() override;

protected:
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  // Constants
  static const double DISPLAY_RESOLUTION_FOR_HITS_MM;  // Bin width of histograms
                                                       // showing hits and tracks
                                                       // (in mm)
  static const double INV_DISPLAY_RESOLUTION_FOR_HITS_MM;

  edm::EDGetTokenT<edm::DetSetVector<TotemTimingDigi>> totemTimingDigiToken_;
  edm::EDGetTokenT<edm::DetSetVector<TotemTimingRecHit>> tokenRecHit_;
  edm::ESGetToken<PPSTimingCalibration, PPSTimingCalibrationRcd> timingCalibrationToken_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;
  unsigned int verbosity_;
  edm::TimeValue_t timeOfPreviousEvent_;

  float verticalShiftBot_, verticalShiftTop_;
  std::unordered_map<unsigned int, double> horizontalShiftOfDiamond_;

  /// plots related to the whole system
  struct GlobalPlots {
    GlobalPlots() {}
    GlobalPlots(DQMStore::IBooker &ibooker);
  };

  GlobalPlots globalPlot_;

  /// plots related to one Diamond detector package
  struct PotPlots {
    // RecHits
    MonitorElement *hitDistribution2d = nullptr;
    MonitorElement *recHitTime = nullptr;

    PotPlots(){};
    PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, PotPlots> potPlots_;

  /// plots related to one Diamond plane
  struct PlanePlots {
    PlanePlots() {}
    PlanePlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, PlanePlots> planePlots_;

  /// plots related to one Diamond channel
  struct ChannelPlots {
    // RecHits
    MonitorElement *recHitTime = nullptr;

    ChannelPlots() {}
    ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id);
  };

  edm::ESHandle<PPSTimingCalibration> hTimingCalib_;
  std::unordered_map<unsigned int, ChannelPlots> channelPlots_;
};

//----------------------------------------------------------------------------------------------------

// Values for all constants
const double DiamondSampicCalibrationDQMSource::DISPLAY_RESOLUTION_FOR_HITS_MM = 0.05;
const double DiamondSampicCalibrationDQMSource::INV_DISPLAY_RESOLUTION_FOR_HITS_MM =
    1. / DISPLAY_RESOLUTION_FOR_HITS_MM;

//----------------------------------------------------------------------------------------------------

DiamondSampicCalibrationDQMSource::GlobalPlots::GlobalPlots(DQMStore::IBooker &ibooker) {
  ibooker.setCurrentFolder("CTPPS/TimingFastSilicon");
}

//----------------------------------------------------------------------------------------------------

DiamondSampicCalibrationDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  CTPPSDiamondDetId(id).rpName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(path);

  CTPPSDiamondDetId(id).rpName(title, CTPPSDiamondDetId::nFull);

  hitDistribution2d = ibooker.book2D("hits in planes",
                                     title + " hits in planes;plane number;x (mm)",
                                     10,
                                     -0.5,
                                     4.5,
                                     19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                     -0.5,
                                     18.5);

  recHitTime = ibooker.book1D("recHit time", title + " time in the recHits; t (ns)", 500, -25, 25);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicCalibrationDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  CTPPSDiamondDetId(id).planeName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(path);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicCalibrationDQMSource::ChannelPlots::ChannelPlots(DQMStore::IBooker &ibooker, unsigned int id) {
  std::string path, title;
  CTPPSDiamondDetId(id).channelName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(path);

  CTPPSDiamondDetId(id).channelName(title, CTPPSDiamondDetId::nFull);
  recHitTime = ibooker.book1D("recHit Time", title + " recHit Time; t (ns)", 500, -25, 25);
}

//----------------------------------------------------------------------------------------------------

DiamondSampicCalibrationDQMSource::DiamondSampicCalibrationDQMSource(const edm::ParameterSet &ps)
    : totemTimingDigiToken_(
          consumes<edm::DetSetVector<TotemTimingDigi>>(ps.getUntrackedParameter<edm::InputTag>("totemTimingDigiTag"))),
      tokenRecHit_(
          consumes<edm::DetSetVector<TotemTimingRecHit>>(ps.getUntrackedParameter<edm::InputTag>("tagRecHits"))),
      timingCalibrationToken_(esConsumes<edm::Transition::BeginRun>()),
      geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      verbosity_(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      timeOfPreviousEvent_(0) {}

//----------------------------------------------------------------------------------------------------

DiamondSampicCalibrationDQMSource::~DiamondSampicCalibrationDQMSource() {}

//----------------------------------------------------------------------------------------------------

void DiamondSampicCalibrationDQMSource::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  // Get detector shifts from the geometry (if present)
  const auto &geom = iSetup.getData(geomEsToken_);
  for (auto it = geom.beginSensor(); it != geom.endSensor(); it++) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);

    const DetGeomDesc *det = geom.sensorNoThrow(detid);
    if (det)
      horizontalShiftOfDiamond_[detid.rpId()] = det->translation().x() - det->getDiamondDimensions().xHalfWidth;
    else
      edm::LogProblem("DiamondSampicCalibrationDQMSource") << "ERROR: no descriptor for detId";
  }
}

//----------------------------------------------------------------------------------------------------

void DiamondSampicCalibrationDQMSource::bookHistograms(DQMStore::IBooker &ibooker,
                                                       const edm::Run &,
                                                       const edm::EventSetup &iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  globalPlot_ = GlobalPlots(ibooker);
  const auto &geom = iSetup.getData(geomEsToken_);
  for (auto it = geom.beginSensor(); it != geom.endSensor(); ++it) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);

    const CTPPSDiamondDetId rpId(detid.arm(), detid.station(), detid.rp());
    potPlots_[rpId] = PotPlots(ibooker, rpId);

    const CTPPSDiamondDetId plId(detid.arm(), detid.station(), detid.rp(), detid.plane());
    planePlots_[plId] = PlanePlots(ibooker, plId);

    const CTPPSDiamondDetId chId(detid.arm(), detid.station(), detid.rp(), detid.plane(), detid.channel());
    channelPlots_[chId] = ChannelPlots(ibooker, chId);
  }
  hTimingCalib_ = iSetup.getHandle(timingCalibrationToken_);
}

//----------------------------------------------------------------------------------------------------

void DiamondSampicCalibrationDQMSource::analyze(const edm::Event &event, const edm::EventSetup &eventSetup) {
  PPSTimingCalibration calib = *hTimingCalib_;
  // get event setup data
  edm::Handle<edm::DetSetVector<TotemTimingRecHit>> timingRecHits;
  event.getByToken(tokenRecHit_, timingRecHits);

  edm::Handle<edm::DetSetVector<TotemTimingDigi>> timingDigi;
  event.getByToken(totemTimingDigiToken_, timingDigi);

  std::unordered_map<uint32_t, uint32_t> detIdToHw;

  for (const auto &digis : *timingDigi) {
    const CTPPSDiamondDetId detId(digis.detId());
    for (const auto &digi : digis)
      detIdToHw[detId] = digi.hardwareId();
  }

  // Using TotemTimingDigi
  std::set<uint8_t> boardSet;
  std::unordered_map<unsigned int, unsigned int> channelsPerPlane;
  std::unordered_map<unsigned int, unsigned int> channelsPerPlaneWithTime;

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
        int numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i)
          potPlots_[detId_pot].hitDistribution2d->Fill(detId.plane() + UFSDShift,
                                                       hitHistoTmpYAxis->GetBinCenter(startBin + i));

        //All plots with Time
        if (rechit.time() != TotemTimingRecHit::NO_T_AVAILABLE) {
          int db = (detIdToHw[detId] & 0xE0) >> 5;
          int sampic = (detIdToHw[detId] & 0x10) >> 4;
          int channel = (detIdToHw[detId] & 0x0F);
          double offset = calib.timeOffset(db, sampic, channel);
          potPlots_[detId_pot].recHitTime->Fill(rechit.time() + offset);
          if (channelPlots_.find(detId) != channelPlots_.end())
            channelPlots_[detId].recHitTime->Fill(rechit.time() + offset);
        }
      }
    }
  }
  // End RecHits

  timeOfPreviousEvent_ = event.time().value();
}

DEFINE_FWK_MODULE(DiamondSampicCalibrationDQMSource);
