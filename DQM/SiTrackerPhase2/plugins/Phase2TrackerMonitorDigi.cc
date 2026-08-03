
// -*- C++ -*-
//
// Package:    Phase2TrackerMonitorDigi
// Class:      Phase2TrackerMonitorDigi
//
/**\class Phase2TrackerMonitorDigi Phase2TrackerMonitorDigi.cc

 Description: It generates various histograms of digi properties. Manual
 switching is enabled for each histogram. Seperate Histograms are there for P
 type and S type sensors of the outer Tracker

*/
//
// Author: Suchandra Dutta, Gourab Saha, Suvankar Roy Chowdhury, Subir Sarkar
// Date: January 29, 2016
// Date: November 8, 2019 (Modified for adding in phase2 DQM Offline)
// Date: July 2026 (Modified by Lisa Juckett for folder restructure)
//

// system include files
#include <memory>

// user includes
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;

class Phase2TrackerMonitorDigi : public DQMEDAnalyzer {
public:
  explicit Phase2TrackerMonitorDigi(const edm::ParameterSet&);
  ~Phase2TrackerMonitorDigi() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  struct DigiMEs {
    MonitorElement* NumberOfDigisPerDet{nullptr};
    MonitorElement* DigiOccupancyP{nullptr};
    MonitorElement* DigiOccupancyS{nullptr};
    MonitorElement* ChargeXYMap{nullptr};
    MonitorElement* ChargeOfDigis{nullptr};
    MonitorElement* ChargeOfDigisVsWidth{nullptr};
    MonitorElement* TotalNumberOfDigisPerLayer{nullptr};
    MonitorElement* NumberOfHitDetectorsPerLayer{nullptr};
    MonitorElement* NumberOfClustersPerDet{nullptr};
    MonitorElement* ClusterWidth{nullptr};
    MonitorElement* FractionOfOvTBits{nullptr};
    MonitorElement* FractionOfOvTBitsVsEta{nullptr};
    MonitorElement* EtaOccupancyProfP{nullptr};
    MonitorElement* EtaOccupancyProfS{nullptr};
    unsigned int nDigiPerLayer{0};
    unsigned int nHitDetsPerLayer{0};
  };

  struct Ph2DigiCluster {
    int charge;
    int position;
    int width;
    int column;
  };

  MonitorElement* XYPositionMap{nullptr};
  MonitorElement* RZPositionMap{nullptr};
  MonitorElement* CrackOverview{nullptr};

private:
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id);
  void fillITPixelDigiHistos(const edm::Handle<edm::DetSetVector<PixelDigi>> handle);
  void fillOTDigiHistos(const edm::Handle<edm::DetSetVector<Phase2TrackerDigi>> handle);
  void fillDigiClusters(DigiMEs& mes, std::vector<Ph2DigiCluster>& digi_clusters);

  const edm::ParameterSet config_;
  std::map<std::string, DigiMEs> layerMEs;
  const bool pixelFlag_;
  const bool clsFlag_;
  const std::string geomType_;
  const edm::InputTag otDigiSrc_;
  const edm::InputTag itPixelDigiSrc_;
  const edm::EDGetTokenT<edm::DetSetVector<Phase2TrackerDigi>> otDigiToken_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> itPixelDigiToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
};

//
// constructors
//
Phase2TrackerMonitorDigi::Phase2TrackerMonitorDigi(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      pixelFlag_(config_.getParameter<bool>("PixelPlotFillingFlag")),
      clsFlag_(config_.getParameter<bool>("StandAloneClusteriserFlag")),
      geomType_(config_.getParameter<std::string>("GeometryType")),
      otDigiSrc_(config_.getParameter<edm::InputTag>("OuterTrackerDigiSource")),
      itPixelDigiSrc_(config_.getParameter<edm::InputTag>("InnerPixelDigiSource")),
      otDigiToken_(consumes<edm::DetSetVector<Phase2TrackerDigi>>(otDigiSrc_)),
      itPixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(itPixelDigiSrc_)),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  LogDebug("Phase2TrackerMonitorDigi") << ">>> Construct Phase2TrackerMonitorDigi ";
}

//
// destructor
//
Phase2TrackerMonitorDigi::~Phase2TrackerMonitorDigi() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  LogDebug("Phase2TrackerMonitorDigi") << ">>> Destroy Phase2TrackerMonitorDigi ";
}

void Phase2TrackerMonitorDigi::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

// -- Analyze
//
void Phase2TrackerMonitorDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get digis
  const auto& pixDigiHandle = iEvent.getHandle(itPixelDigiToken_);
  const auto& otDigiHandle = iEvent.getHandle(otDigiToken_);

  // Tracker Topology
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    if (pixelFlag_)
      fillITPixelDigiHistos(pixDigiHandle);
    else
      fillOTDigiHistos(otDigiHandle);
  }
}
void Phase2TrackerMonitorDigi::fillITPixelDigiHistos(const edm::Handle<edm::DetSetVector<PixelDigi>> handle) {
  const edm::DetSetVector<PixelDigi>* digis = handle.product();

  for (typename edm::DetSetVector<PixelDigi>::const_iterator DSViter = digis->begin(); DSViter != digis->end();
       DSViter++) {
    unsigned int rawid = DSViter->id;
    LogDebug("Phase2TrackerMonitorDigi") << " Det Id = " << rawid;

    int layer = tTopo_->getITPixelLayerNumber(rawid);

    if (layer < 0)
      continue;
    const DetId detId(rawid);
    if (DetId(detId).det() != DetId::Detector::Tracker)
      continue;

    const GeomDetUnit* gDetUnit = tkGeom_->idToDetUnit(detId);
    const GeomDet* geomDet = tkGeom_->idToDet(detId);

    GlobalPoint detPos = geomDet->surface().toGlobal(Local2DPoint(0, 0));
    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(gDetUnit);
    int nRows = tkDetUnit->specificTopology().nrows();
    int nColumns = tkDetUnit->specificTopology().ncolumns();
    if (nRows * nColumns == 0)
      continue;

    int nDigi = 0;
    int row_last = -1;
    int col_last = -1;
    std::vector<Ph2DigiCluster> digiClusters;
    for (typename edm::DetSet<PixelDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      int col = di->column();  // column
      int row = di->row();     // row
      int adc = di->adc();     // digi charge
      if (geomDet) {
        MeasurementPoint mp(row + 0.5, col + 0.5);
        GlobalPoint pdPos = geomDet->surface().toGlobal(gDetUnit->topology().localPosition(mp));
        if (XYPositionMap)
          XYPositionMap->Fill(pdPos.x(), pdPos.y());
        if (RZPositionMap)
          RZPositionMap->Fill(pdPos.z(), std::hypot(pdPos.x(), pdPos.y()));
      }
      nDigi++;
      LogDebug("Phase2TrackerMonitorDigi") << "  column " << col << " row " << row << std::dec << std::endl;
      if (clsFlag_) {
        if (row_last == -1 || abs(row - row_last) != 1 || col != col_last) {
          Ph2DigiCluster dClus;
          dClus.position = row + 1;
          dClus.column = col;
          dClus.width = 1;
          dClus.charge = 255;
          digiClusters.push_back(dClus);
        } else {
          int pos = digiClusters.back().position + row + 1;
          int width = digiClusters.back().width + 1;
          pos /= width;

          digiClusters.back().position = pos;
          digiClusters.back().width += 1;
        }
        LogDebug("Phase2TrackerMonitorDigi") << " row " << row << " col " << col << " row_last " << row_last
                                             << " col_last " << col_last << " width " << digiClusters.back().width;
        row_last = row;
        col_last = col;
      }

      for (int fillingDepth = 1; fillingDepth <= 6; fillingDepth++) {
        std::string key = phase2tkutil::getHistoId(detId, tTopo_, detPos.phi(), fillingDepth, false);
        std::map<std::string, DigiMEs>::iterator pos = layerMEs.find(key);

        if (pos == layerMEs.end())
          continue;
        DigiMEs& local_mes = pos->second;

        local_mes.nDigiPerLayer++;

        if (clsFlag_)
          fillDigiClusters(local_mes, digiClusters);

        if (local_mes.ChargeXYMap)
          local_mes.ChargeXYMap->Fill(col, row, adc);

        if (local_mes.ChargeOfDigis)
          local_mes.ChargeOfDigis->Fill(adc);

        if (nDigi == int(DSViter->size())) {
          // Reached the end of digis in this det
          // Fill local histos that should only be filled once per det
          local_mes.nHitDetsPerLayer++;
          if (local_mes.NumberOfDigisPerDet)
            local_mes.NumberOfDigisPerDet->Fill(nDigi);
          float occupancy = 1.0;
          if (nRows * nColumns > 0)
            occupancy = nDigi * 1.0 / (nRows * nColumns);
          if (geomDet) {
            GlobalPoint gp = geomDet->surface().toGlobal(
                gDetUnit->topology().localPosition(MeasurementPoint(nRows / 2.0, nColumns / 2.0)));
            if (local_mes.EtaOccupancyProfP)
              local_mes.EtaOccupancyProfP->Fill(gp.eta(), occupancy);
          }

          if (local_mes.DigiOccupancyP)
            local_mes.DigiOccupancyP->Fill(occupancy);
        }
      }
    }
  }
  // Fill histograms after loop over digis are complete
  for (auto& ilayer : layerMEs) {
    DigiMEs& local_mes = ilayer.second;
    if (local_mes.TotalNumberOfDigisPerLayer)
      local_mes.TotalNumberOfDigisPerLayer->Fill(local_mes.nDigiPerLayer);
    if (local_mes.NumberOfHitDetectorsPerLayer)
      local_mes.NumberOfHitDetectorsPerLayer->Fill(local_mes.nHitDetsPerLayer);
    local_mes.nDigiPerLayer = 0;
    local_mes.nHitDetsPerLayer = 0;
  }
}
void Phase2TrackerMonitorDigi::fillOTDigiHistos(const edm::Handle<edm::DetSetVector<Phase2TrackerDigi>> handle) {
  const edm::DetSetVector<Phase2TrackerDigi>* digis = handle.product();

  for (typename edm::DetSetVector<Phase2TrackerDigi>::const_iterator DSViter = digis->begin(); DSViter != digis->end();
       DSViter++) {
    // Loop over dets
    unsigned int rawid = DSViter->id;
    DetId detId(rawid);
    LogDebug("Phase2TrackerMonitorDigi") << " Det Id = " << rawid;
    int layer = tTopo_->getOTLayerNumber(rawid);
    if (layer < 0)
      continue;
    if (DetId(detId).det() != DetId::Detector::Tracker)
      continue;

    const GeomDetUnit* gDetUnit = tkGeom_->idToDetUnit(detId);
    const GeomDet* geomDet = tkGeom_->idToDet(detId);
    GlobalPoint detPos = geomDet->surface().toGlobal(Local2DPoint(0, 0));

    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(gDetUnit);
    int module = tTopo_->module(detId);
    // CRACK is viewed from behind, so to align plots with what is seen in real life, modules are flipped
    if (CrackOverview)
      module = std::abs(int(module - 13));
    TrackerGeometry::ModuleType moduleType = tkGeom_->getDetectorType(DetId(detId));
    int nRows = tkDetUnit->specificTopology().nrows();
    int nColumns = tkDetUnit->specificTopology().ncolumns();
    if (nRows * nColumns == 0)
      continue;

    int nDigi = 0;
    int row_last = -1;
    int col_last = -1;
    float over_threshold = 0;
    float frac_ot = 0.;
    std::vector<Ph2DigiCluster> digiClusters;

    for (typename edm::DetSet<Phase2TrackerDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      // Loop over digis in a specific det
      int col = di->column();  // column
      int row = di->row();     // row
      const DetId detId(rawid);

      nDigi++;
      if (di->overThreshold())
        over_threshold++;
      LogDebug("Phase2TrackerMonitorDigi") << "  column " << col << " row " << row << std::dec << std::endl;

      if (clsFlag_) {
        if (row_last == -1 || abs(row - row_last) != 1 || col != col_last) {
          Ph2DigiCluster dClus;
          dClus.position = row + 1;
          dClus.column = col;
          dClus.width = 1;
          dClus.charge = 255;
          digiClusters.push_back(dClus);
        } else {
          int pos = digiClusters.back().position + row + 1;
          int width = digiClusters.back().width + 1;
          pos /= width;

          digiClusters.back().position = pos;
          digiClusters.back().width += 1;
        }
        row_last = row;
        col_last = col;
        LogDebug("Phase2TrackerMonitorDigi") << " row " << row << " col " << col << " row_last " << row_last
                                             << " col_last " << col_last << " width " << digiClusters.back().width;
      }

      // Fill non-layer histograms
      if (geomDet) {
        MeasurementPoint mp(row + 0.5, col + 0.5);
        GlobalPoint pdPos = geomDet->surface().toGlobal(gDetUnit->topology().localPosition(mp));
        if (XYPositionMap)
          XYPositionMap->Fill(pdPos.x(), pdPos.y());
        if (RZPositionMap)
          RZPositionMap->Fill(pdPos.z(), std::hypot(pdPos.x(), pdPos.y()));
      }
      if (CrackOverview)
        CrackOverview->Fill(module, layer + 0.05 - (module % 2 * 0.1));

      for (int fillingDepth = 1; fillingDepth <= 6; fillingDepth++) {
        std::string key = phase2tkutil::getHistoId(detId, tTopo_, detPos.phi(), fillingDepth, false);
        std::map<std::string, DigiMEs>::iterator pos = layerMEs.find(key);
        if (pos == layerMEs.end())
          continue;
        DigiMEs& local_mes = pos->second;

        local_mes.nDigiPerLayer++;

        if (clsFlag_)
          fillDigiClusters(local_mes, digiClusters);

        if (nDigi == int(DSViter->size())) {
          // Reached the end of digis in this det
          // Fill local histos that should only be filled once per det
          local_mes.nHitDetsPerLayer++;
          if (local_mes.NumberOfDigisPerDet)
            local_mes.NumberOfDigisPerDet->Fill(nDigi);

          if (nDigi > 0)
            frac_ot = over_threshold / nDigi;
          if (local_mes.FractionOfOvTBits && moduleType == TrackerGeometry::ModuleType::Ph2PSS)
            local_mes.FractionOfOvTBits->Fill(frac_ot);

          float occupancy = 1.0;
          if (nRows * nColumns > 0)
            occupancy = nDigi * 1.0 / (nRows * nColumns);
          if (geomDet) {
            GlobalPoint gp =
                geomDet->surface().toGlobal(gDetUnit->topology().localPosition(MeasurementPoint(0.0, 0.0)));
            if (moduleType == TrackerGeometry::ModuleType::Ph2PSP) {
              if (local_mes.DigiOccupancyP)
                local_mes.DigiOccupancyP->Fill(occupancy);
              if (local_mes.EtaOccupancyProfP)
                local_mes.EtaOccupancyProfP->Fill(gp.eta(), occupancy);
            } else {
              if (local_mes.DigiOccupancyS)
                local_mes.DigiOccupancyS->Fill(occupancy);
              if (local_mes.EtaOccupancyProfS)
                local_mes.EtaOccupancyProfS->Fill(gp.eta(), occupancy);
              if (local_mes.FractionOfOvTBitsVsEta && moduleType == TrackerGeometry::ModuleType::Ph2PSS)
                local_mes.FractionOfOvTBitsVsEta->Fill(gp.eta(), frac_ot);
            }
          }
        }
      }
    }
  }
  // Fill histograms after loop over digis are complete
  for (auto& ilayer : layerMEs) {
    DigiMEs& local_mes = ilayer.second;
    if (local_mes.TotalNumberOfDigisPerLayer)
      local_mes.TotalNumberOfDigisPerLayer->Fill(local_mes.nDigiPerLayer);
    if (local_mes.NumberOfHitDetectorsPerLayer)
      local_mes.NumberOfHitDetectorsPerLayer->Fill(local_mes.nHitDetsPerLayer);
    local_mes.nDigiPerLayer = 0;
    local_mes.nHitDetsPerLayer = 0;
  }
}
//
// -- Book Histograms
//
void Phase2TrackerMonitorDigi::bookHistograms(DQMStore::IBooker& ibooker,
                                              edm::Run const& iRun,
                                              edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;

  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker, detId_raw);
    }
  }
  ibooker.cd();
  std::stringstream folder_name;
  folder_name << top_folder;
  ibooker.setCurrentFolder(folder_name.str() + "/Positions");

  XYPositionMap = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("XYPositionMapH"), ibooker);
  RZPositionMap = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("RZPositionMapH"), ibooker);

  ibooker.setCurrentFolder(folder_name.str());

  edm::ParameterSet Parameters = config_.getParameter<edm::ParameterSet>("CrackOverview");
  if (Parameters.getParameter<bool>("switch")) {
    CrackOverview = ibooker.book2DPoly(Parameters.getParameter<std::string>("name"),
                                       Parameters.getParameter<std::string>("title"),
                                       Parameters.getParameter<double>("xmin"),
                                       Parameters.getParameter<double>("xmax"),
                                       Parameters.getParameter<double>("ymin"),
                                       Parameters.getParameter<double>("ymax"));
    if (CrackOverview->getTH2Poly()->GetNumberOfBins() == 0) {
      double yOffset = 0;
      for (int layer = 1; layer < 7; layer++) {
        for (int module = 1; module < 13; module++) {
          if (module % 2 == 1)
            yOffset = -0.1;
          else
            yOffset = 0;
          CrackOverview->addBin(module - 0.7, layer + yOffset, module + 0.7, layer + yOffset + 0.1);
        }
      }
    }
    CrackOverview->getTH2Poly()->SetStats(false);
    CrackOverview->setOption("z0");

  } else
    CrackOverview = nullptr;
}
//
// -- Book Layer Histograms
//
void Phase2TrackerMonitorDigi::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id) {
  int layer;
  if (pixelFlag_)
    layer = tTopo_->getITPixelLayerNumber(det_id);
  else
    layer = tTopo_->getOTLayerNumber(det_id);

  if (layer < 0)
    return;

  const GeomDet* geomDet = tkGeom_->idToDet(det_id);
  GlobalPoint detPos = geomDet->surface().toGlobal(Local2DPoint(0, 0));
  TrackerGeometry::ModuleType moduleType = tkGeom_->getDetectorType(DetId(det_id));
  for (int bookingDepth = 1; bookingDepth <= 6; bookingDepth++) {
    std::string key = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, true);
    std::map<std::string, DigiMEs>::iterator pos = layerMEs.find(key);

    if (pos == layerMEs.end()) {
      std::string top_folder = config_.getParameter<std::string>("TopFolderName");
      std::stringstream folder_name;

      // For endCap: P-type sensors are present only upto ring 10 for discs 1&2
      // (TEDD-1) and upto ring 7 for discs 3,4&5 (TEDD-2)
      bool isPtypeSensor = (pixelFlag_ || moduleType == TrackerGeometry::ModuleType::Ph2PSP);

      ibooker.cd();
      ibooker.setCurrentFolder(top_folder + "/" + key);
      LogDebug("Phase2TrackerMonitorDigi") << " Booking Histograms in : " << key;

      DigiMEs local_mes;

      local_mes.NumberOfDigisPerDet = phase2tkutil::book1DFromPSetWithPosition(
          config_.getParameter<edm::ParameterSet>("NumberOfDigisPerDetH"), ibooker, prettyName, false);
      local_mes.TotalNumberOfDigisPerLayer = phase2tkutil::book1DFromPSetWithPosition(
          config_.getParameter<edm::ParameterSet>("TotalNumberOfDigisPerLayerH"), ibooker, prettyName, false);
      local_mes.NumberOfHitDetectorsPerLayer = phase2tkutil::book1DFromPSetWithPosition(
          config_.getParameter<edm::ParameterSet>("NumberOfHitDetsPerLayerH"), ibooker, prettyName, false);

      // Plots only for the inner pixel
      if (pixelFlag_) {
        local_mes.ChargeXYMap = phase2tkutil::book2DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("ChargeXYMapH"), ibooker, prettyName);
        local_mes.ChargeOfDigis = phase2tkutil::book1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("DigiChargeH"), ibooker, prettyName, false);
        // For standalone clusteriser
        if (clsFlag_)
          local_mes.ChargeOfDigisVsWidth = phase2tkutil::book2DFromPSetWithPosition(
              config_.getParameter<edm::ParameterSet>("DigiChargeVsWidthH"), ibooker, prettyName);
      }
      // For outer tracker modules (S-type histograms)
      else {
        local_mes.DigiOccupancyS = phase2tkutil::book1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("DigiOccupancySH"), ibooker, prettyName, false);
        local_mes.EtaOccupancyProfS = phase2tkutil::bookProfile1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("DigiOccupancyVsEtaSH"), ibooker, prettyName);

        // FracOfOverThresholdBits is only available for S-type sensor of PS module
        // For booking, just check a PS sensor exists. Must be filled in PS-S only
        if (isPtypeSensor) {
          local_mes.FractionOfOvTBits = phase2tkutil::book1DFromPSetWithPosition(
              config_.getParameter<edm::ParameterSet>("DigisOverThresholdH"), ibooker, prettyName, false);
          local_mes.FractionOfOvTBitsVsEta = phase2tkutil::bookProfile1DFromPSetWithPosition(
              config_.getParameter<edm::ParameterSet>("DigiFractionOverThresholdVsEtaH"), ibooker, prettyName);
        }
      }

      // Plots for P-type sensor (Pixel or P-side of PS module)
      if (isPtypeSensor) {
        local_mes.DigiOccupancyP = phase2tkutil::book1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("DigiOccupancyPH"), ibooker, prettyName, false);
        local_mes.EtaOccupancyProfP = phase2tkutil::bookProfile1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("DigiOccupancyVsEtaPH"), ibooker, prettyName);
      }

      // Plots for Standalone clusters (Can be switched on from configs)
      if (clsFlag_) {
        local_mes.NumberOfClustersPerDet = phase2tkutil::book1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("NumberOfClustersPerDetH"), ibooker, prettyName, false);
        local_mes.ClusterWidth = phase2tkutil::book1DFromPSetWithPosition(
            config_.getParameter<edm::ParameterSet>("ClusterWidthH"), ibooker, prettyName, false);
      }

      layerMEs.insert(std::make_pair(key, local_mes));
    }
  }
}

void Phase2TrackerMonitorDigi::fillDigiClusters(DigiMEs& mes, std::vector<Ph2DigiCluster>& digi_clusters) {
  int nclus = digi_clusters.size();
  if (mes.NumberOfClustersPerDet)
    mes.NumberOfClustersPerDet->Fill(nclus);
  for (auto& iclus : digi_clusters) {
    if (mes.ClusterWidth)
      mes.ClusterWidth->Fill(iclus.width);
    if (pixelFlag_ && mes.ChargeOfDigisVsWidth)
      mes.ChargeOfDigisVsWidth->Fill(iclus.charge, iclus.width);
  }
}
// define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerMonitorDigi);
