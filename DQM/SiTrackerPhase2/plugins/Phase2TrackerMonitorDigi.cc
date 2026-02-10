
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
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
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
  std::string getHistoId(uint32_t det_id, bool flag);

  struct DigiMEs {
    MonitorElement* NumberOfDigisPerDet{nullptr};
    MonitorElement* DigiOccupancyP{nullptr};
    MonitorElement* DigiOccupancyS{nullptr};
    MonitorElement* ChargeXYMap{nullptr};
    MonitorElement* PositionOfDigisP{nullptr};
    MonitorElement* PositionOfDigisS{nullptr};
    MonitorElement* ChargeOfDigis{nullptr};
    MonitorElement* ChargeOfDigisVsWidth{nullptr};
    MonitorElement* TotalNumberOfDigisPerLayer{nullptr};
    MonitorElement* NumberOfHitDetectorsPerLayer{nullptr};
    MonitorElement* NumberOfClustersPerDet{nullptr};
    MonitorElement* ClusterWidth{nullptr};
    MonitorElement* ClusterPositionP{nullptr};
    MonitorElement* ClusterPositionS{nullptr};
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
  MonitorElement* XYOccupancyMap{nullptr};
  MonitorElement* RZOccupancyMap{nullptr};
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

    std::string key = getHistoId(rawid, pixelFlag_);
    std::map<std::string, DigiMEs>::iterator pos = layerMEs.find(key);

    if (pos == layerMEs.end())
      continue;

    if (DetId(detId).det() != DetId::Detector::Tracker)
      continue;

    const GeomDetUnit* gDetUnit = tkGeom_->idToDetUnit(detId);
    const GeomDet* geomDet = tkGeom_->idToDet(detId);

    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(gDetUnit);
    int nRows = tkDetUnit->specificTopology().nrows();
    int nColumns = tkDetUnit->specificTopology().ncolumns();
    if (nRows * nColumns == 0)
      continue;

    DigiMEs& local_mes = pos->second;

    local_mes.nHitDetsPerLayer++;

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
      if (local_mes.ChargeXYMap)
        local_mes.ChargeXYMap->Fill(col, row, adc);
      if (local_mes.PositionOfDigisP)
        local_mes.PositionOfDigisP->Fill(row + 1, col + 1);
      if (local_mes.ChargeOfDigis)
        local_mes.ChargeOfDigis->Fill(adc);
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
    }
    if (local_mes.NumberOfDigisPerDet)
      local_mes.NumberOfDigisPerDet->Fill(nDigi);
    if (clsFlag_)
      fillDigiClusters(local_mes, digiClusters);
    local_mes.nDigiPerLayer += nDigi;
    float occupancy = 1.0;
    if (nRows * nColumns > 0)
      occupancy = nDigi * 1.0 / (nRows * nColumns);
    if (geomDet) {
      GlobalPoint gp = geomDet->surface().toGlobal(
          gDetUnit->topology().localPosition(MeasurementPoint(nRows / 2.0, nColumns / 2.0)));
      if (XYOccupancyMap)
        XYOccupancyMap->Fill(gp.x(), gp.y(), occupancy);
      if (RZOccupancyMap)
        RZOccupancyMap->Fill(gp.z(), std::hypot(gp.x(), gp.y()), occupancy);
      if (local_mes.EtaOccupancyProfP)
        local_mes.EtaOccupancyProfP->Fill(gp.eta(), occupancy);
    }

    if (local_mes.DigiOccupancyP)
      local_mes.DigiOccupancyP->Fill(occupancy);
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
    unsigned int rawid = DSViter->id;
    DetId detId(rawid);
    LogDebug("Phase2TrackerMonitorDigi") << " Det Id = " << rawid;
    int layer = tTopo_->getOTLayerNumber(rawid);
    if (layer < 0)
      continue;
    std::string key = getHistoId(rawid, pixelFlag_);
    std::map<std::string, DigiMEs>::iterator pos = layerMEs.find(key);
    if (pos == layerMEs.end())
      continue;
    DigiMEs& local_mes = pos->second;

    local_mes.nHitDetsPerLayer++;
    if (DetId(detId).det() != DetId::Detector::Tracker)
      continue;

    const GeomDetUnit* gDetUnit = tkGeom_->idToDetUnit(detId);
    const GeomDet* geomDet = tkGeom_->idToDet(detId);

    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(gDetUnit);
    int nRows = tkDetUnit->specificTopology().nrows();
    int nColumns = tkDetUnit->specificTopology().ncolumns();
    if (nRows * nColumns == 0)
      continue;

    int nDigi = 0;
    int row_last = -1;
    int col_last = -1;
    float frac_ot = 0.;
    std::vector<Ph2DigiCluster> digiClusters;
    for (typename edm::DetSet<Phase2TrackerDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
      int col = di->column();  // column
      int row = di->row();     // row
      const DetId detId(rawid);

      if (geomDet) {
        MeasurementPoint mp(row + 0.5, col + 0.5);
        GlobalPoint pdPos = geomDet->surface().toGlobal(gDetUnit->topology().localPosition(mp));
        if (XYPositionMap)
          XYPositionMap->Fill(pdPos.x(), pdPos.y());
        if (RZPositionMap)
          RZPositionMap->Fill(pdPos.z(), std::hypot(pdPos.x(), pdPos.y()));
      }
      nDigi++;
      if (di->overThreshold())
        frac_ot++;
      LogDebug("Phase2TrackerMonitorDigi") << "  column " << col << " row " << row << std::dec << std::endl;
      if (nColumns > 2 && local_mes.PositionOfDigisP)
        local_mes.PositionOfDigisP->Fill(row + 1, col + 1);
      if (nColumns <= 2 && local_mes.PositionOfDigisS)
        local_mes.PositionOfDigisS->Fill(row + 1, col + 1);
      if (CrackOverview)
        CrackOverview->Fill(tTopo_->module(rawid), layer - 0.05 + (tTopo_->module(rawid) % 2 * 0.1));

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
    }
    if (local_mes.NumberOfDigisPerDet)
      local_mes.NumberOfDigisPerDet->Fill(nDigi);
    if (clsFlag_)
      fillDigiClusters(local_mes, digiClusters);
    local_mes.nDigiPerLayer += nDigi;
    if (nDigi)
      frac_ot /= nDigi;
    if (local_mes.FractionOfOvTBits && nColumns <= 2)
      local_mes.FractionOfOvTBits->Fill(frac_ot);

    float occupancy = 1.0;
    if (nRows * nColumns > 0)
      occupancy = nDigi * 1.0 / (nRows * nColumns);
    if (geomDet) {
      GlobalPoint gp = geomDet->surface().toGlobal(gDetUnit->topology().localPosition(MeasurementPoint(0.0, 0.0)));
      if (XYOccupancyMap)
        XYOccupancyMap->Fill(gp.x(), gp.y(), occupancy);
      if (RZOccupancyMap)
        RZOccupancyMap->Fill(gp.z(), std::hypot(gp.x(), gp.y()), occupancy);
      if (nColumns > 2) {
        if (local_mes.DigiOccupancyP)
          local_mes.DigiOccupancyP->Fill(occupancy);
        if (local_mes.EtaOccupancyProfP)
          local_mes.EtaOccupancyProfP->Fill(gp.eta(), occupancy);
      } else {
        if (local_mes.DigiOccupancyS)
          local_mes.DigiOccupancyS->Fill(occupancy);
        if (local_mes.EtaOccupancyProfS)
          local_mes.EtaOccupancyProfS->Fill(gp.eta(), occupancy);
        if (local_mes.FractionOfOvTBitsVsEta)
          local_mes.FractionOfOvTBitsVsEta->Fill(gp.eta(), frac_ot);
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
  folder_name << top_folder << "/"
              << "DigiMonitor";
  ibooker.setCurrentFolder(folder_name.str());

  edm::ParameterSet Parameters = config_.getParameter<edm::ParameterSet>("XYPositionMapH");
  edm::ParameterSet ParametersOcc = config_.getParameter<edm::ParameterSet>("DigiOccupancyPH");
  if (Parameters.getParameter<bool>("switch")) {
    XYPositionMap = ibooker.book2D("Digi_Position_XY",
                                   "Digi_Position_XY",
                                   Parameters.getParameter<int32_t>("Nxbins"),
                                   Parameters.getParameter<double>("xmin"),
                                   Parameters.getParameter<double>("xmax"),
                                   Parameters.getParameter<int32_t>("Nybins"),
                                   Parameters.getParameter<double>("ymin"),
                                   Parameters.getParameter<double>("ymax"));
    XYPositionMap->setAxisTitle("Digi position X [cm]", 1);
    XYPositionMap->setAxisTitle("Digi position Y [cm]", 2);
  } else
    XYPositionMap = nullptr;

  if (Parameters.getParameter<bool>("switch") && ParametersOcc.getParameter<bool>("switch")) {
    XYOccupancyMap = ibooker.bookProfile2D("Digi_Occupancy_XY",
                                           "Digi_Occupancy_XY",
                                           Parameters.getParameter<int32_t>("Nxbins"),
                                           Parameters.getParameter<double>("xmin"),
                                           Parameters.getParameter<double>("xmax"),
                                           Parameters.getParameter<int32_t>("Nybins"),
                                           Parameters.getParameter<double>("ymin"),
                                           Parameters.getParameter<double>("ymax"),
                                           ParametersOcc.getParameter<double>("xmin"),
                                           ParametersOcc.getParameter<double>("xmax"));
    XYOccupancyMap->setAxisTitle("Digi position X [cm]", 1);
    XYOccupancyMap->setAxisTitle("Digi position Y [cm]", 2);
  } else
    XYOccupancyMap = nullptr;

  Parameters = config_.getParameter<edm::ParameterSet>("RZPositionMapH");
  if (Parameters.getParameter<bool>("switch")) {
    RZPositionMap = ibooker.book2D("Digi_Position_RZ",
                                   "Digi_Position_RZ",
                                   Parameters.getParameter<int32_t>("Nxbins"),
                                   Parameters.getParameter<double>("xmin"),
                                   Parameters.getParameter<double>("xmax"),
                                   Parameters.getParameter<int32_t>("Nybins"),
                                   Parameters.getParameter<double>("ymin"),
                                   Parameters.getParameter<double>("ymax"));
    RZPositionMap->setAxisTitle("Digi position z [cm]", 1);
    RZPositionMap->setAxisTitle("Digi position #rho [cm]", 2);
  } else
    RZPositionMap = nullptr;

  if (Parameters.getParameter<bool>("switch") && ParametersOcc.getParameter<bool>("switch")) {
    RZOccupancyMap = ibooker.bookProfile2D("Digi_Occupancy_RZ",
                                           "Digi_Occupancy_RZ",
                                           Parameters.getParameter<int32_t>("Nxbins"),
                                           Parameters.getParameter<double>("xmin"),
                                           Parameters.getParameter<double>("xmax"),
                                           Parameters.getParameter<int32_t>("Nybins"),
                                           Parameters.getParameter<double>("ymin"),
                                           Parameters.getParameter<double>("ymax"),
                                           ParametersOcc.getParameter<double>("xmin"),
                                           ParametersOcc.getParameter<double>("xmax"));
    RZOccupancyMap->setAxisTitle("Digi position z [cm]", 1);
    RZOccupancyMap->setAxisTitle("Digi position #rho [cm]", 2);
  } else
    RZOccupancyMap = nullptr;

  Parameters = config_.getParameter<edm::ParameterSet>("CrackOverview");
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
          if (module % 2 == 0)
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
  std::string key = getHistoId(det_id, pixelFlag_);
  std::map<std::string, DigiMEs>::iterator pos = layerMEs.find(key);

  if (pos == layerMEs.end()) {
    std::string top_folder = config_.getParameter<std::string>("TopFolderName");
    std::stringstream folder_name;

    // For endCap: P-type sensors are present only upto ring 10 for discs 1&2
    // (TEDD-1) and upto ring 7 for discs 3,4&5 (TEDD-2)
    bool isPStypeModForTEDD_1 =
        (!pixelFlag_ && layer > 100 && tTopo_->tidWheel(det_id) < 3 && tTopo_->tidRing(det_id) <= 10) ? true : false;
    bool isPStypeModForTEDD_2 =
        (!pixelFlag_ && layer > 100 && tTopo_->tidWheel(det_id) >= 3 && tTopo_->tidRing(det_id) <= 7) ? true : false;

    bool isPtypeSensor =
        (pixelFlag_ || (layer < 4 || (layer > 6 && (isPStypeModForTEDD_1 || isPStypeModForTEDD_2)))) ? true : false;

    ibooker.cd();
    ibooker.setCurrentFolder(top_folder + "/DigiMonitor/" + key);
    LogDebug("Phase2TrackerMonitorDigi") << " Booking Histograms in : " << key;

    std::ostringstream HistoName;
    DigiMEs local_mes;

    edm::ParameterSet Parameters = config_.getParameter<edm::ParameterSet>("NumberOfDigisPerDetH");
    edm::ParameterSet EtaParameters = config_.getParameter<edm::ParameterSet>("EtaH");
    HistoName.str("");
    HistoName << "NumberOfDigisPerDet";
    if (Parameters.getParameter<bool>("switch"))
      local_mes.NumberOfDigisPerDet = ibooker.book1D(HistoName.str(),
                                                     HistoName.str(),
                                                     Parameters.getParameter<int32_t>("Nbins"),
                                                     Parameters.getParameter<double>("xmin"),
                                                     Parameters.getParameter<double>("xmax"));

    Parameters = config_.getParameter<edm::ParameterSet>("TotalNumberOfDigisPerLayerH");
    HistoName.str("");
    HistoName << "TotalNumberOfDigisPerLayer";
    if (Parameters.getParameter<bool>("switch"))
      local_mes.TotalNumberOfDigisPerLayer = ibooker.book1D(HistoName.str(),
                                                            HistoName.str(),
                                                            Parameters.getParameter<int32_t>("Nbins"),
                                                            Parameters.getParameter<double>("xmin"),
                                                            Parameters.getParameter<double>("xmax"));

    Parameters = config_.getParameter<edm::ParameterSet>("NumberOfHitDetsPerLayerH");
    HistoName.str("");
    HistoName << "NumberOfHitDetectorsPerLayer";
    if (Parameters.getParameter<bool>("switch"))
      local_mes.NumberOfHitDetectorsPerLayer = ibooker.book1D(HistoName.str(),
                                                              HistoName.str(),
                                                              Parameters.getParameter<int32_t>("Nbins"),
                                                              Parameters.getParameter<double>("xmin"),
                                                              Parameters.getParameter<double>("xmax"));

    // Plots only for the inner pixel
    if (pixelFlag_) {
      Parameters = config_.getParameter<edm::ParameterSet>("ChargeXYMapH");
      if (Parameters.getParameter<bool>("switch")) {
        local_mes.ChargeXYMap = ibooker.book2D("Digi_Charge_XY",
                                               "Digi_Charge_XY",
                                               Parameters.getParameter<int32_t>("Nxbins"),
                                               Parameters.getParameter<double>("xmin"),
                                               Parameters.getParameter<double>("xmax"),
                                               Parameters.getParameter<int32_t>("Nybins"),
                                               Parameters.getParameter<double>("ymin"),
                                               Parameters.getParameter<double>("ymax"));
        local_mes.ChargeXYMap->setAxisTitle("Row", 1);
        local_mes.ChargeXYMap->setAxisTitle("Column", 2);
      }
      Parameters = config_.getParameter<edm::ParameterSet>("DigiChargeH");
      if (Parameters.getParameter<bool>("switch")) {
        local_mes.ChargeOfDigis = ibooker.book1D("Digi_Charge",
                                                 "Digi_Charge",
                                                 Parameters.getParameter<int32_t>("Nbins"),
                                                 Parameters.getParameter<double>("xmin"),
                                                 Parameters.getParameter<double>("xmax"));
        local_mes.ChargeOfDigis->setAxisTitle("Digi charge [ADC]", 1);
      }
      // For standalone clusteriser
      if (clsFlag_) {
        edm::ParameterSet WidthParameters = config_.getParameter<edm::ParameterSet>("ClusterWidthH");
        if (Parameters.getParameter<bool>("switch") && WidthParameters.getParameter<bool>("switch")) {
          local_mes.ChargeOfDigisVsWidth = ibooker.book2D("Digi_Charge_vs_Width",
                                                          "Digi_Charge_vs_Width",
                                                          Parameters.getParameter<int32_t>("Nbins"),
                                                          Parameters.getParameter<double>("xmin"),
                                                          Parameters.getParameter<double>("xmax"),
                                                          WidthParameters.getParameter<int32_t>("Nbins"),
                                                          WidthParameters.getParameter<double>("xmin"),
                                                          WidthParameters.getParameter<double>("xmax"));
          local_mes.ChargeOfDigisVsWidth->setAxisTitle("Digi charge [ADC]", 1);
          local_mes.ChargeOfDigisVsWidth->setAxisTitle("Digi width", 2);
        }
      }
    }
    // For outer tracker modules (S-type histograms)
    else {
      Parameters = config_.getParameter<edm::ParameterSet>("DigiOccupancySH");
      if (Parameters.getParameter<bool>("switch"))
        local_mes.DigiOccupancyS = ibooker.book1D("Digi_Occupancy_S",
                                                  "Digi_Occupancy_S",
                                                  Parameters.getParameter<int32_t>("Nbins"),
                                                  Parameters.getParameter<double>("xmin"),
                                                  Parameters.getParameter<double>("xmax"));

      if (Parameters.getParameter<bool>("switch") && EtaParameters.getParameter<bool>("switch")) {
        local_mes.EtaOccupancyProfS = ibooker.bookProfile("Digi_Occupancy_vs_eta_S",
                                                          "Digi_Occupancy_vs_eta_S",
                                                          EtaParameters.getParameter<int32_t>("Nbins"),
                                                          EtaParameters.getParameter<double>("xmin"),
                                                          EtaParameters.getParameter<double>("xmax"),
                                                          Parameters.getParameter<double>("xmin"),
                                                          Parameters.getParameter<double>("xmax"),
                                                          "");
        local_mes.EtaOccupancyProfS->setAxisTitle("#eta", 1);
      }

      // choose the correct PSet by module type
      if (isPtypeSensor) {
        Parameters = config_.getParameter<edm::ParameterSet>("PositionOfDigisSHPS");
      } else {
        Parameters = config_.getParameter<edm::ParameterSet>("PositionOfDigisSH2S");
      }
      if (Parameters.getParameter<bool>("switch")) {
        local_mes.PositionOfDigisS = ibooker.book2D("Digi_Position_S",
                                                    "Digi_Position_S",
                                                    Parameters.getParameter<int32_t>("Nxbins"),
                                                    Parameters.getParameter<double>("xmin"),
                                                    Parameters.getParameter<double>("xmax"),
                                                    Parameters.getParameter<int32_t>("Nybins"),
                                                    Parameters.getParameter<double>("ymin"),
                                                    Parameters.getParameter<double>("ymax"));
        local_mes.PositionOfDigisS->setAxisTitle("Strip", 1);
        local_mes.PositionOfDigisS->setAxisTitle("Half-sensor", 2);
      }

      // For standalone clusteriser
      if (clsFlag_) {
        // Choose the correct PSet by module type
        if (isPtypeSensor) {
          Parameters = config_.getParameter<edm::ParameterSet>("ClusterPositionSHPS");
        } else {
          Parameters = config_.getParameter<edm::ParameterSet>("ClusterPositionSH2S");
        }

        if (Parameters.getParameter<bool>("switch")) {
          local_mes.ClusterPositionS = ibooker.book2D("Cluster_Position_S",
                                                      "Cluster_Position_S",
                                                      Parameters.getParameter<int32_t>("Nxbins"),
                                                      Parameters.getParameter<double>("xmin"),
                                                      Parameters.getParameter<double>("xmax"),
                                                      Parameters.getParameter<int32_t>("Nybins"),
                                                      Parameters.getParameter<double>("ymin"),
                                                      Parameters.getParameter<double>("ymax"));
          local_mes.ClusterPositionS->setAxisTitle("Strip", 1);
          local_mes.ClusterPositionS->setAxisTitle("Half-sensor", 2);
        }
      }
      // Only for the S-type sensor of PS module
      // FracOfOverThresholdBits is only available for S-type sensor of PS
      // module
      if (isPStypeModForTEDD_1 || isPStypeModForTEDD_2) {
        local_mes.FractionOfOvTBits =
            ibooker.book1D("Digis_FractionOverThreshold", "Digis_FractionOverThreshold", 11, -0.05, 1.05);

        Parameters = config_.getParameter<edm::ParameterSet>("NumberOfDigisPerDetH");
        if (Parameters.getParameter<bool>("switch") && EtaParameters.getParameter<bool>("switch")) {
          local_mes.FractionOfOvTBitsVsEta = ibooker.bookProfile("Digis_FractionOverThreshold_vs_eta",
                                                                 "Digis_FractionOverThreshold_vs_eta",
                                                                 EtaParameters.getParameter<int32_t>("Nbins"),
                                                                 EtaParameters.getParameter<double>("xmin"),
                                                                 EtaParameters.getParameter<double>("xmax"),
                                                                 Parameters.getParameter<double>("xmin"),
                                                                 Parameters.getParameter<double>("xmax"),
                                                                 "");
          local_mes.FractionOfOvTBitsVsEta->setAxisTitle("#eta", 1);
        }
      }
    }

    // Plots for P-type sensor (Pixel or P-side of PS module)
    if (isPtypeSensor) {
      Parameters = config_.getParameter<edm::ParameterSet>("DigiOccupancyPH");
      if (Parameters.getParameter<bool>("switch"))
        local_mes.DigiOccupancyP = ibooker.book1D("Digi_Occupancy_P",
                                                  "Digi_Occupancy_P",
                                                  Parameters.getParameter<int32_t>("Nbins"),
                                                  Parameters.getParameter<double>("xmin"),
                                                  Parameters.getParameter<double>("xmax"));

      HistoName.str("");
      HistoName << "DigiOccupancyVsEtaP";
      if (Parameters.getParameter<bool>("switch") && EtaParameters.getParameter<bool>("switch")) {
        local_mes.EtaOccupancyProfP = ibooker.bookProfile("Digi_Occupancy_vs_eta_P",
                                                          "Digi_Occupancy_vs_eta_P",
                                                          EtaParameters.getParameter<int32_t>("Nbins"),
                                                          EtaParameters.getParameter<double>("xmin"),
                                                          EtaParameters.getParameter<double>("xmax"),
                                                          Parameters.getParameter<double>("xmin"),
                                                          Parameters.getParameter<double>("xmax"),
                                                          "");
        local_mes.EtaOccupancyProfP->setAxisTitle("#eta", 1);
      }

      Parameters = config_.getParameter<edm::ParameterSet>("PositionOfDigisPH");
      if (Parameters.getParameter<bool>("switch")) {
        local_mes.PositionOfDigisP = ibooker.book2D("Digi_Position_P",
                                                    "Digi_Position_P",
                                                    Parameters.getParameter<int32_t>("Nxbins"),
                                                    Parameters.getParameter<double>("xmin"),
                                                    Parameters.getParameter<double>("xmax"),
                                                    Parameters.getParameter<int32_t>("Nybins"),
                                                    Parameters.getParameter<double>("ymin"),
                                                    Parameters.getParameter<double>("ymax"));
        local_mes.PositionOfDigisP->setAxisTitle("Column", 1);
        local_mes.PositionOfDigisP->setAxisTitle("Row", 2);
      }

      if (clsFlag_) {
        Parameters = config_.getParameter<edm::ParameterSet>("ClusterPositionPH");
        if (Parameters.getParameter<bool>("switch")) {
          local_mes.ClusterPositionP = ibooker.book2D("Cluster_Position_P",
                                                      "Cluster_Position_P",
                                                      Parameters.getParameter<int32_t>("Nxbins"),
                                                      Parameters.getParameter<double>("xmin"),
                                                      Parameters.getParameter<double>("xmax"),
                                                      Parameters.getParameter<int32_t>("Nybins"),
                                                      Parameters.getParameter<double>("ymin"),
                                                      Parameters.getParameter<double>("ymax"));
          local_mes.ClusterPositionP->setAxisTitle("Column", 1);
          local_mes.ClusterPositionP->setAxisTitle("Row", 2);
        }
      }
    }

    // Plots for Standalone clusters (Can be switched on from configs)
    if (clsFlag_) {
      Parameters = config_.getParameter<edm::ParameterSet>("NumberOfClustersPerDetH");
      HistoName.str("");
      HistoName << "NumberOfClustersPerDet";
      if (Parameters.getParameter<bool>("switch"))
        local_mes.NumberOfClustersPerDet = ibooker.book1D(HistoName.str(),
                                                          HistoName.str(),
                                                          Parameters.getParameter<int32_t>("Nbins"),
                                                          Parameters.getParameter<double>("xmin"),
                                                          Parameters.getParameter<double>("xmax"));
      Parameters = config_.getParameter<edm::ParameterSet>("ClusterWidthH");
      if (Parameters.getParameter<bool>("switch")) {
        local_mes.ClusterWidth = ibooker.book1D("Cluster_Width",
                                                "Cluster_Width",
                                                Parameters.getParameter<int32_t>("Nbins"),
                                                Parameters.getParameter<double>("xmin"),
                                                Parameters.getParameter<double>("xmax"));
        local_mes.ClusterWidth->setAxisTitle("Cluster width", 1);
      }
    }
    layerMEs.insert(std::make_pair(key, local_mes));
  }
}

std::string Phase2TrackerMonitorDigi::getHistoId(uint32_t det_id, bool flag) {
  if (flag)
    return phase2tkutil::getITHistoId(det_id, tTopo_);
  else
    return phase2tkutil::getOTHistoId(det_id, tTopo_);
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
    if (mes.ClusterPositionP)
      mes.ClusterPositionP->Fill(iclus.position, iclus.column + 1);
    if (!pixelFlag_ && mes.ClusterPositionS && iclus.column <= 2)
      mes.ClusterPositionS->Fill(iclus.position, iclus.column + 1);
  }
}
// define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerMonitorDigi);
