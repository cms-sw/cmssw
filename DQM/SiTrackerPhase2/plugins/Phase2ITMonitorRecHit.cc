// Package:    Phase2ITMonitorRecHit
// Class:      Phase2ITMonitorRecHit
//
/**\class Phase2ITMonitorRecHit Phase2ITMonitorRecHit.cc 
 Description:  Plugin for Phase2 RecHit validation
*/
//
// Author: Shubhi Parolia, Suvankar Roy Chowdhury
// Date: July 2020
//
// system include files
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
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

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
  MonitorElement* numberRecHits_;
  MonitorElement* globalXY_barrel_;
  MonitorElement* globalXY_endcap_;
  MonitorElement* globalRZ_barrel_;
  MonitorElement* globalRZ_endcap_;

  struct RecHitME {
    MonitorElement* numberRecHits = nullptr;
    MonitorElement* globalPosXY = nullptr;
    MonitorElement* globalPosRZ = nullptr;
    MonitorElement* localPosXY = nullptr;
    MonitorElement* posX = nullptr;
    MonitorElement* posY = nullptr;
    MonitorElement* poserrX = nullptr;
    MonitorElement* poserrY = nullptr;
    MonitorElement* clusterSizeX = nullptr;
    MonitorElement* clusterSizeY = nullptr;
  };
  std::map<std::string, RecHitME> layerMEs_;
};

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
  edm::Handle<SiPixelRecHitCollection> rechits;
  iEvent.getByToken(tokenRecHitsIT_, rechits);
  if (!rechits.isValid())
    return;
  std::map<std::string, unsigned int> nrechitLayerMap;
  unsigned long int nTotrechitsinevt = 0;
  // Loop over modules
  for (const auto& DSViter : *rechits) {
    // Get the detector id
    unsigned int rawid(DSViter.detId());
    DetId detId(rawid);
    // Get the geomdet
    const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(detId));
    if (!geomDetunit)
      continue;
    std::string key = phase2tkutil::getITHistoId(detId.rawId(), tTopo_);
    nTotrechitsinevt += DSViter.size();
    if (nrechitLayerMap.find(key) == nrechitLayerMap.end()) {
      nrechitLayerMap.emplace(key, DSViter.size());
    } else {
      nrechitLayerMap[key] += DSViter.size();
    }

    //loop over rechits for a single detId
    for (const auto& rechit : DSViter) {
      LocalPoint lp = rechit.localPosition();
      Global3DPoint globalPos = geomDetunit->surface().toGlobal(lp);
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
      //layer wise histo
      if (layerMEs_[key].clusterSizeX)
        layerMEs_[key].clusterSizeX->Fill(rechit.cluster()->sizeX());
      if (layerMEs_[key].clusterSizeY)
        layerMEs_[key].clusterSizeY->Fill(rechit.cluster()->sizeY());
      if (layerMEs_[key].globalPosXY)
        layerMEs_[key].globalPosXY->Fill(gx, gy);
      if (layerMEs_[key].globalPosRZ)
        layerMEs_[key].globalPosRZ->Fill(gz, gr);
      if (layerMEs_[key].localPosXY)
        layerMEs_[key].localPosXY->Fill(lp.x(), lp.y());
      if (layerMEs_[key].posX)
        layerMEs_[key].posX->Fill(lp.x());
      if (layerMEs_[key].posY)
        layerMEs_[key].posY->Fill(lp.y());
      float eta = geomDetunit->surface().toGlobal(lp).eta();
      if (layerMEs_[key].poserrX)
        layerMEs_[key].poserrX->Fill(eta, million * rechit.localPositionError().xx());
      if (layerMEs_[key].poserrY)
        layerMEs_[key].poserrY->Fill(eta, million * rechit.localPositionError().yy());
    }  //end loop over rechits of a detId
  }    //End loop over DetSetVector

  //fill nRecHits per event
  numberRecHits_->Fill(nTotrechitsinevt);
  //fill nRecHit counter per layer
  for (const auto& lme : nrechitLayerMap)
    if (layerMEs_[lme.first].numberRecHits)
      layerMEs_[lme.first].numberRecHits->Fill(lme.second);
}

void Phase2ITMonitorRecHit::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerGeometry> geomHandle = iSetup.getHandle(geomToken_);
  tkGeom_ = &(*geomHandle);
  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(topoToken_);
  tTopo_ = tTopoHandle.product();
}

void Phase2ITMonitorRecHit::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  ibooker.cd();
  edm::LogInfo("Phase2ITMonitorRecHit") << " Booking Histograms in : " << top_folder;
  std::string dir = top_folder;
  ibooker.setCurrentFolder(dir);
  std::stringstream HistoName;
  //Global histos for IT
  HistoName.str("");
  HistoName << "NumberRecHits";
  numberRecHits_ = phase2tkutil::book1DFromPSet(
      config_.getParameter<edm::ParameterSet>("GlobalNumberRecHits"), HistoName.str(), ibooker);
  HistoName.str("");
  HistoName << "Global_Position_XY_IT_barrel";
  globalXY_barrel_ = phase2tkutil::book2DFromPSet(
      config_.getParameter<edm::ParameterSet>("GlobalPositionXY_PXB"), HistoName.str(), ibooker);
  HistoName.str("");
  HistoName << "Global_Position_RZ_IT_barrel";
  globalRZ_barrel_ = phase2tkutil::book2DFromPSet(
      config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_PXB"), HistoName.str(), ibooker);
  HistoName.str("");
  HistoName << "Global_Position_XY_IT_endcap";
  globalXY_endcap_ = phase2tkutil::book2DFromPSet(
      config_.getParameter<edm::ParameterSet>("GlobalPositionXY_PXEC"), HistoName.str(), ibooker);
  HistoName.str("");
  HistoName << "Global_Position_RZ_IT_endcap";
  globalRZ_endcap_ = phase2tkutil::book2DFromPSet(
      config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_PXEC"), HistoName.str(), ibooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    edm::ESHandle<TrackerGeometry> geomHandle = iSetup.getHandle(geomToken_);
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (!(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
            det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      edm::LogInfo("Phase2ITMonitorRecHit") << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
                                            << "\t key=" << phase2tkutil::getITHistoId(detId_raw, tTopo_) << std::endl;
      bookLayerHistos(ibooker, detId_raw, dir);
    }
  }
}
// -- Book Layer Histograms
void Phase2ITMonitorRecHit::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  std::string key = phase2tkutil::getITHistoId(det_id, tTopo_);
  if (key.empty())
    return;
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    std::ostringstream histoName;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2ITMonitorRecHit") << " Booking Histograms in : " << (subdir + "/" + key);
    histoName.str("");
    histoName << "Number_RecHits";
    local_histos.numberRecHits = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("LocalNumberRecHits"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "RecHit_X";
    local_histos.posX =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("RecHitPosX"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "RecHit_Y";
    local_histos.posY =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("RecHitPosY"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "RecHit_X_error_Vs_eta";
    local_histos.poserrX = phase2tkutil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("RecHitPosErrorX_Eta"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "RecHit_Y_error_Vs_eta";
    local_histos.poserrY = phase2tkutil::bookProfile1DFromPSet(
        config_.getParameter<edm::ParameterSet>("RecHitPosErrorY_Eta"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "Cluster_SizeX";
    local_histos.clusterSizeX = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("LocalClusterSizeX"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "Cluster_SizeY";
    local_histos.clusterSizeY = phase2tkutil::book1DFromPSet(
        config_.getParameter<edm::ParameterSet>("LocalClusterSizeY"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "Global_Position_XY";
    local_histos.globalPosXY = phase2tkutil::book2DFromPSet(
        config_.getParameter<edm::ParameterSet>("GlobalPositionXY_perlayer"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "Global_Position_RZ";
    local_histos.globalPosRZ = phase2tkutil::book2DFromPSet(
        config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_perlayer"), histoName.str(), ibooker);

    histoName.str("");
    histoName << "Local_Position_XY";
    local_histos.localPosXY = phase2tkutil::book2DFromPSet(
        config_.getParameter<edm::ParameterSet>("LocalPositionXY"), histoName.str(), ibooker);
    layerMEs_.emplace(key, local_histos);
  }
}

void Phase2ITMonitorRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // rechitMonitorIT
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberRecHits");
    psd0.add<std::string>("title", "NumberRecHits;Number of RecHits;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.0);
    psd0.add<int>("NxBins", 50);
    desc.add<edm::ParameterSetDescription>("GlobalNumberRecHits", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_Position_RZ_IT_barrel");
    psd0.add<std::string>("title", "Global_Position_RZ_IT_barrel;z [mm];r [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 1500);
    psd0.add<int>("NyBins", 300);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_PXB", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_Position_XY_IT_barrel");
    psd0.add<std::string>("title", "Global_Position_XY_IT_barrel;x [mm];y [mm];");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 600);
    psd0.add<int>("NyBins", 600);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 300.0);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("ymin", -300.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_PXB", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_Position_RZ_IT_endcap");
    psd0.add<std::string>("title", "Global_Position_RZ_IT_endcap;z [mm];r [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 1500);
    psd0.add<int>("NyBins", 300);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_PXEC", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_Position_XY_IT_endcap");
    psd0.add<std::string>("title", "Global_Position_XY_IT_endcap; x [mm]; y [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 600);
    psd0.add<int>("NyBins", 600);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 300.0);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("ymin", -300.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_PXEC", psd0);
  }
  //Per layer/ring histos
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalNumberRecHits");
    psd0.add<std::string>("title", "NumberRecHits;Number of RecHits;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.0);
    psd0.add<int>("NxBins", 50);
    desc.add<edm::ParameterSetDescription>("LocalNumberRecHits", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_Position_RZ");
    psd0.add<std::string>("title", "Global_Position_RZ;z [mm];r [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 1500);
    psd0.add<int>("NyBins", 300);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_perlayer", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_Position_XY");
    psd0.add<std::string>("title", "Global_Position_XY;x [mm]; y[mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 600);
    psd0.add<int>("NyBins", 600);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 300.0);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("ymin", -300.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Local_Position_XY");
    psd0.add<std::string>("title", "Local_Position_XY; x; y");
    psd0.add<double>("ymax", 0.0);
    psd0.add<int>("NxBins", 500);
    psd0.add<int>("NyBins", 500);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.0);
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_SizeX");
    psd0.add<std::string>("title", "Cluster_SizeX; cluster size y;");
    psd0.add<double>("xmin", -0.5);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 20.5);
    psd0.add<int>("NxBins", 21);
    desc.add<edm::ParameterSetDescription>("LocalClusterSizeX", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_SizeY");
    psd0.add<std::string>("title", "Cluster_SizeY;cluster size y;");
    psd0.add<double>("xmin", -0.5);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 20.5);
    psd0.add<int>("NxBins", 21);
    desc.add<edm::ParameterSetDescription>("LocalClusterSizeY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "RecHit_X");
    psd0.add<std::string>("title", "RecHit_X;RecHit position X dimension;");
    psd0.add<double>("xmin", -2.5);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 2.5);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("RecHitPosX", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "RecHit_Y");
    psd0.add<std::string>("title", "RecHit_Y;RecHit position X dimension;");
    psd0.add<double>("xmin", -2.5);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 2.5);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("RecHitPosY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "RecHit_X_error_Vs_eta");
    psd0.add<std::string>("title", "RecHit_X_error_Vs_eta;#eta;x error #times 10^{6}");
    psd0.add<bool>("switch", true);
    psd0.add<int>("NxBins", 82);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymax", 10.0);
    psd0.add<double>("ymin", 0.);
    desc.add<edm::ParameterSetDescription>("RecHitPosErrorX_Eta", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "RecHit_Y_error_Vs_eta");
    psd0.add<std::string>("title", "RecHit_Y_error_Vs_eta;#eta;y error #times 10^{6}");
    psd0.add<bool>("switch", true);
    psd0.add<int>("NxBins", 82);
    psd0.add<double>("xmax", 4.1);
    psd0.add<double>("xmin", -4.1);
    psd0.add<double>("ymax", 10.0);
    psd0.add<double>("ymin", 0.);
    desc.add<edm::ParameterSetDescription>("RecHitPosErrorY_Eta", psd0);
  }
  desc.add<std::string>("TopFolderName", "TrackerPhase2ITRecHit");
  desc.add<edm::InputTag>("rechitsSrc", edm::InputTag("siPixelRecHits"));
  descriptions.add("Phase2ITMonitorRecHit", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(Phase2ITMonitorRecHit);
