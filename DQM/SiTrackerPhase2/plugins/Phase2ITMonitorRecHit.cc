// Package:    Phase2ITMonitorRecHit
// Class:      Phase2ITMonitorRecHit
//
/**\class Phase2ITMonitorRecHit Phase2ITMonitorRecHit.cc 
 Description:  Plugin for Phase2 RecHit validation
*/
//
// Author: Shubhi Parolia, Suvankar Roy Chowdhury
// Date: July 2020
// Date: August 2026 (Modified by Lisa Juckett for folder restructure)
#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/TrackerGeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class Phase2ITMonitorRecHit : public DQMEDAnalyzer {
public:
  explicit Phase2ITMonitorRecHit(const edm::ParameterSet&);
  ~Phase2ITMonitorRecHit() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void fillITHistos(const edm::Event& iEvent);
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

  edm::ParameterSet config_;
  std::string geomType_;
  const edm::EDGetTokenT<SiPixelRecHitCollection> tokenRecHitsIT_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
  static constexpr float million = 1e6;
  MonitorElement* globalXY_barrel_;
  MonitorElement* globalXY_endcap_;
  MonitorElement* globalRZ_barrel_;
  MonitorElement* globalRZ_endcap_;

  struct RecHitME {
    MonitorElement* numberRecHits = nullptr;
    MonitorElement* posX = nullptr;
    MonitorElement* posY = nullptr;
    MonitorElement* poserrX = nullptr;
    MonitorElement* poserrY = nullptr;
    MonitorElement* clusterSizeX = nullptr;
    MonitorElement* clusterSizeY = nullptr;
    unsigned int recHitCounter;
  };
  std::map<std::string, RecHitME> layerMEs_;
  enum Level { IT = 1, SUBSTRUCTURE, SHELL, ENDCAP_RING, ENDCAP_WHEEL, LAYER };
};
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

Phase2ITMonitorRecHit::Phase2ITMonitorRecHit(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      tokenRecHitsIT_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("rechitsSrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2ITMonitorRecHit") << ">>> Construct Phase2ITMonitorRecHit ";
}

Phase2ITMonitorRecHit::~Phase2ITMonitorRecHit() {
  edm::LogInfo("Phase2ITMonitorRecHit") << ">>> Destroy Phase2ITMonitorRecHit ";
}
// -- Analyze
void Phase2ITMonitorRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) { fillITHistos(iEvent); }

void Phase2ITMonitorRecHit::fillITHistos(const edm::Event& iEvent) {
  // Get the RecHits
  const auto& rechits = iEvent.getHandle(tokenRecHitsIT_);
  if (!rechits.isValid())
    return;
  // Loop over modules
  for (const auto& DSViter : *rechits) {
    // Get the detector id
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    // Get the geomdet
    const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(detId));
    if (!geomDetunit)
      continue;

    GlobalPoint detPos = geomDetunit->surface().toGlobal(Local2DPoint(0, 0));
    //loop over rechits for a single detId
    for (const auto& rechit : DSViter) {
      LocalPoint lp = rechit.localPosition();
      Global3DPoint globalPos = geomDetunit->surface().toGlobal(lp);
      float eta = geomDetunit->surface().toGlobal(lp).eta();
      //in mm
      double gx = globalPos.x() * 10.;
      double gy = globalPos.y() * 10.;
      double gz = globalPos.z() * 10.;
      double gr = globalPos.perp() * 10.;
      //Fill global positions
      if (geomDetunit->subDetector() == GeomDetEnumerators::SubDetector::P2PXB) {
        globalXY_barrel_->Fill(gx, gy);
        globalRZ_barrel_->Fill(gz, gr);
      } else if (geomDetunit->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC) {
        globalXY_endcap_->Fill(gx, gy);
        globalRZ_endcap_->Fill(gz, gr);
      }
      for (enum Level fillingDepth = IT; fillingDepth <= LAYER; fillingDepth = Level(fillingDepth + 1)) {
        // Skip filling for barrel detIds on endcap-only depths
        if ((fillingDepth == ENDCAP_RING || fillingDepth == ENDCAP_WHEEL) &&
            DetId(detId).subdetId() == PixelSubdetector::PixelBarrel)
          continue;
        std::string key = phase2tkutil::getHistoId(detId.rawId(), tTopo_, detPos.phi(), fillingDepth, false);

        if (layerMEs_[key].clusterSizeX)
          layerMEs_[key].clusterSizeX->Fill(rechit.cluster()->sizeX());
        if (layerMEs_[key].clusterSizeY)
          layerMEs_[key].clusterSizeY->Fill(rechit.cluster()->sizeY());
        if (layerMEs_[key].posX)
          layerMEs_[key].posX->Fill(lp.x());
        if (layerMEs_[key].posY)
          layerMEs_[key].posY->Fill(lp.y());
        if (layerMEs_[key].poserrX)
          layerMEs_[key].poserrX->Fill(eta, million * rechit.localPositionError().xx());
        if (layerMEs_[key].poserrY)
          layerMEs_[key].poserrY->Fill(eta, million * rechit.localPositionError().yy());
        layerMEs_[key].recHitCounter++;
      }  // End layer ME filling loop
    }  //end loop over rechits of a detId
  }  //End loop over DetSetVector
  //fill nRecHit counter per layer
  for (auto& lme : layerMEs_) {
    RecHitME& local_mes = lme.second;
    if (local_mes.numberRecHits)
      local_mes.numberRecHits->Fill(local_mes.recHitCounter);
    local_mes.recHitCounter = 0;
  }
}

void Phase2ITMonitorRecHit::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

void Phase2ITMonitorRecHit::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  ibooker.cd();
  std::string dir = top_folder;
  ibooker.setCurrentFolder(dir + "/Positions");
  edm::LogInfo("Phase2ITMonitorRecHit") << " Booking Histograms in : " << top_folder << "/Positions";
  globalXY_barrel_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_PXB"), ibooker);

  globalRZ_barrel_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_PXB"), ibooker);

  globalXY_endcap_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_PXEC"), ibooker);

  globalRZ_endcap_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_PXEC"), ibooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (!(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
            det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      GlobalPoint detPos = det_u->surface().toGlobal(Local2DPoint(0, 0));
      edm::LogInfo("Phase2ITMonitorRecHit")
          << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
          << "\t key=" << phase2tkutil::getHistoId(detId_raw, tTopo_, detPos.phi(), LAYER, false) << std::endl;
      bookLayerHistos(ibooker, detId_raw, dir);
    }
  }
}
// -- Book Layer Histograms
void Phase2ITMonitorRecHit::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  const GeomDetUnit* geomDetUnit = tkGeom_->idToDetUnit(det_id);
  GlobalPoint detPos = geomDetUnit->surface().toGlobal(Local2DPoint(0, 0));
  for (enum Level bookingDepth = IT; bookingDepth <= LAYER; bookingDepth = Level(bookingDepth + 1)) {
    // Skip booking for barrel detIds on endcap-only depths
    if ((bookingDepth == ENDCAP_RING || bookingDepth == ENDCAP_WHEEL) &&
        DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel)
      continue;

    std::string key = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, true);

    if (layerMEs_.find(key) == layerMEs_.end()) {
      ibooker.cd();
      RecHitME local_histos;
      ibooker.setCurrentFolder(subdir + "/" + key);
      edm::LogInfo("Phase2ITMonitorRecHit") << " Booking Histograms in : " << (subdir + "/" + key);

      local_histos.numberRecHits = phase2tkutil::book1DFromPSet(
          config_.getParameter<edm::ParameterSet>("LocalNumberRecHits"), ibooker, prettyName, bookingDepth);

      local_histos.posX =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("RecHitPosX"), ibooker, prettyName);

      local_histos.posY =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("RecHitPosY"), ibooker, prettyName);

      local_histos.poserrX = phase2tkutil::bookProfile1DFromPSet(
          config_.getParameter<edm::ParameterSet>("RecHitPosErrorX_Eta"), ibooker, prettyName);

      local_histos.poserrY = phase2tkutil::bookProfile1DFromPSet(
          config_.getParameter<edm::ParameterSet>("RecHitPosErrorY_Eta"), ibooker, prettyName);

      local_histos.clusterSizeX = phase2tkutil::book1DFromPSet(
          config_.getParameter<edm::ParameterSet>("LocalClusterSizeX"), ibooker, prettyName);

      local_histos.clusterSizeY = phase2tkutil::book1DFromPSet(
          config_.getParameter<edm::ParameterSet>("LocalClusterSizeY"), ibooker, prettyName);

      layerMEs_.emplace(key, local_histos);
    }
  }
}

void Phase2ITMonitorRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  phase2tkutil::add1DDesc(
      desc, "GlobalNumberRecHits", "Num_RecHits", "NumberRecHits", "Number of RecHits", "", 250, 0.0, 250000.0);

  // Positions
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_PXB",
                          "RecHit_Global_Position_RZ_IT_barrel",
                          "RecHit_Global_Position_RZ_IT_barrel",
                          "z [mm]",
                          "r [mm]",
                          1500,
                          -3000.0,
                          3000.0,
                          300,
                          0.0,
                          300.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_PXB",
                          "RecHit_Global_Position_XY_IT_barrel",
                          "RecHit_Global_Position_XY_IT_barrel",
                          "x [mm]",
                          "y [mm]",
                          600,
                          -300.0,
                          300.0,
                          600,
                          -300.0,
                          300.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_PXEC",
                          "RecHit_Global_Position_RZ_IT_endcap",
                          "RecHit_Global_Position_RZ_IT_endcap",
                          "z [mm]",
                          "r [mm]",
                          1500,
                          -3000.0,
                          3000.0,
                          300,
                          0.0,
                          300.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_PXEC",
                          "RecHit_Global_Position_XY_IT_endcap",
                          "RecHit_Global_Position_XY_IT_endcap",
                          "x [mm]",
                          "y [mm]",
                          600,
                          -300.0,
                          300.0,
                          600,
                          -300.0,
                          300.0);

  // Per layer/ring histos
  phase2tkutil::add1DDesc(desc,
                          "LocalNumberRecHits",
                          "Num_RecHits_Per_Event",
                          "Number of RecHits per event in {}",
                          "Number of RecHits",
                          "Number of events",
                          150,
                          0.0,
                          250000.0);
  phase2tkutil::add1DDesc(desc,
                          "LocalClusterSizeX",
                          "RecHit_Size_X",
                          "RecHit size in X dimension in {}",
                          "RecHit size x",
                          "Number of RecHits",
                          21,
                          -0.5,
                          20.5);
  phase2tkutil::add1DDesc(desc,
                          "LocalClusterSizeY",
                          "RecHit_Size_Y",
                          "RecHit size in Y dimension in {}",
                          "RecHit size y",
                          "Number of RecHits",
                          26,
                          -0.5,
                          25.5);
  phase2tkutil::add1DDesc(desc,
                          "RecHitPosX",
                          "RecHit_X",
                          "RecHit position in X dimension in {}",
                          "RecHit position X dimension",
                          "Number of RecHits",
                          100,
                          -2.5,
                          2.5);
  phase2tkutil::add1DDesc(desc,
                          "RecHitPosY",
                          "RecHit_Y",
                          "RecHit position in Y dimension in {}",
                          "RecHit position Y dimension",
                          "Number of RecHits",
                          100,
                          -2.5,
                          2.5);

  // 1DProfiles - 2D desc with NyBins = 0
  phase2tkutil::add2DDesc(desc,
                          "RecHitPosErrorX_Eta",
                          "RecHit_X_error_Vs_eta",
                          "RecHit X error Vs eta in {}",
                          "#eta",
                          "x error [#mum]",
                          82,
                          -4.1,
                          4.1,
                          0,
                          0.0,
                          10.0);
  phase2tkutil::add2DDesc(desc,
                          "RecHitPosErrorY_Eta",
                          "RecHit_Y_error_Vs_eta",
                          "RecHit Y error Vs eta in {}",
                          "#eta",
                          "y error [#mum]",
                          82,
                          -4.1,
                          4.1,
                          0,
                          0.0,
                          10.0);

  desc.add<std::string>("TopFolderName", "InnerTracker");
  desc.add<edm::InputTag>("rechitsSrc", edm::InputTag("siPixelRecHits"));
  descriptions.add("Phase2ITMonitorRecHit", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2ITMonitorRecHit);
