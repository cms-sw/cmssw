// Package:    Phase2OTMonitorRecHit
// Class:      Phase2OTMonitorRecHit
//
/**\class Phase2OTMonitorRecHit Phase2OTMonitorRecHit.cc 
 Description:  Standalone  Plugin for Phase2 RecHit validation
*/
//
// Author: Suvankar Roy Chowdhury
// Date: March 2021
//
// system include files
#include <memory>
#include <map>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2OTMonitorRecHit : public DQMEDAnalyzer {
public:
  explicit Phase2OTMonitorRecHit(const edm::ParameterSet&);
  ~Phase2OTMonitorRecHit() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir);

  edm::ParameterSet config_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> tokenRecHitsOT_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;

  MonitorElement* numberRecHits_ = nullptr;
  MonitorElement* globalXY_P_ = nullptr;
  MonitorElement* globalRZ_P_ = nullptr;
  MonitorElement* globalXY_S_ = nullptr;
  MonitorElement* globalRZ_S_ = nullptr;
  struct RecHitME {
    // use TH1D instead of TH1F to avoid stauration at 2^31
    // above this increments with +1 don't work for float, need double
    MonitorElement* globalPosXY_P = nullptr;
    MonitorElement* globalPosXY_S = nullptr;
    MonitorElement* localPosXY_P = nullptr;
    MonitorElement* localPosXY_S = nullptr;

    MonitorElement* numberRecHits_P = nullptr;
    MonitorElement* numberRecHits_S = nullptr;
    MonitorElement* clusterSize_P = nullptr;
    MonitorElement* clusterSize_S = nullptr;
  };
  std::map<std::string, RecHitME> layerMEs_;
};

//
// constructors
//
Phase2OTMonitorRecHit::Phase2OTMonitorRecHit(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      tokenRecHitsOT_(consumes<Phase2TrackerRecHit1DCollectionNew>(config_.getParameter<edm::InputTag>("rechitsSrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2OTMonitorRecHit") << ">>> Construct Phase2OTMonitorRecHit ";
}

//
// destructor
//
Phase2OTMonitorRecHit::~Phase2OTMonitorRecHit() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  edm::LogInfo("Phase2OTMonitorRecHit") << ">>> Destroy Phase2OTMonitorRecHit ";
}
//
// -- DQM Begin Run
void Phase2OTMonitorRecHit::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
  ;
}

//
// -- Analyze
//
void Phase2OTMonitorRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the RecHits Phase2TrackerRecHit1DCollectionNew
  const auto& rechits = iEvent.getHandle(tokenRecHitsOT_);
  if (!rechits.isValid())
    return;
  std::map<std::string, unsigned int> nrechitLayerMapP;
  std::map<std::string, unsigned int> nrechitLayerMapS;
  unsigned long int nTotrechitsinevt = 0;
  // Loop over modules
  Phase2TrackerRecHit1DCollectionNew::const_iterator DSViter;
  for (DSViter = rechits->begin(); DSViter != rechits->end(); ++DSViter) {
    // Get the detector unit's id
    unsigned int rawid(DSViter->detId());
    DetId detId(rawid);
    // Get the geomdet
    const GeomDetUnit* geomDetunit(tkGeom_->idToDetUnit(detId));
    if (!geomDetunit)
      continue;
    // determine the detector we are in
    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
    std::string key = phase2tkutil::getOTHistoId(detId.rawId(), tTopo_);
    nTotrechitsinevt += DSViter->size();
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      if (nrechitLayerMapP.find(key) == nrechitLayerMapP.end()) {
        nrechitLayerMapP.insert(std::make_pair(key, DSViter->size()));
      } else {
        nrechitLayerMapP[key] += DSViter->size();
      }
    } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
      if (nrechitLayerMapS.find(key) == nrechitLayerMapS.end()) {
        nrechitLayerMapS.insert(std::make_pair(key, DSViter->size()));
      } else {
        nrechitLayerMapS[key] += DSViter->size();
      }
    }
    edmNew::DetSet<Phase2TrackerRecHit1D>::const_iterator rechitIt;
    //loop over rechits for a single detId
    for (rechitIt = DSViter->begin(); rechitIt != DSViter->end(); ++rechitIt) {
      LocalPoint lp = rechitIt->localPosition();
      Global3DPoint globalPos = geomDetunit->surface().toGlobal(lp);
      //in mm
      double gx = globalPos.x() * 10.;
      double gy = globalPos.y() * 10.;
      double gz = globalPos.z() * 10.;
      double gr = globalPos.perp() * 10.;
      //Fill positions
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        globalXY_P_->Fill(gx, gy);
        globalRZ_P_->Fill(gz, gr);
        //layer wise histo
        layerMEs_[key].clusterSize_P->Fill(rechitIt->cluster()->size());
        //layerMEs_[key].globalPosXY_P->Fill(gx, gy);
        layerMEs_[key].localPosXY_P->Fill(lp.x(), lp.y());
      } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
        globalXY_S_->Fill(gx, gy);
        globalRZ_S_->Fill(gz, gr);
        //layer wise histo
        layerMEs_[key].clusterSize_S->Fill(rechitIt->cluster()->size());
        //layerMEs_[key].globalPosXY_S->Fill(gx, gy);
        layerMEs_[key].localPosXY_S->Fill(lp.x(), lp.y());
      }
    }  //end loop over rechits of a detId
  }    //End loop over DetSetVector

  //fill nRecHits per event
  numberRecHits_->Fill(nTotrechitsinevt);
  //fill nRecHit counter per layer
  for (auto& lme : nrechitLayerMapP) {
    layerMEs_[lme.first].numberRecHits_P->Fill(lme.second);
  }
  for (auto& lme : nrechitLayerMapS) {
    layerMEs_[lme.first].numberRecHits_S->Fill(lme.second);
  }
}
//
// -- Book Histograms
//
void Phase2OTMonitorRecHit::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  //std::stringstream folder_name;

  ibooker.cd();
  edm::LogInfo("Phase2OTMonitorRecHit") << " Booking Histograms in : " << top_folder;
  ibooker.setCurrentFolder(top_folder);

  //Global histos for OT
  numberRecHits_ = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalNRecHits"), ibooker);

  globalXY_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_P"), ibooker);

  globalRZ_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_P"), ibooker);

  globalXY_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_S"), ibooker);

  globalRZ_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_S"), ibooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
          det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC)
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//
// -- Book Layer Histograms
//
void Phase2OTMonitorRecHit::bookLayerHistos(DQMStore::IBooker& ibooker, unsigned int det_id, std::string& subdir) {
  std::string key = phase2tkutil::getOTHistoId(det_id, tTopo_);
  if (layerMEs_.find(key) == layerMEs_.end()) {
    ibooker.cd();
    RecHitME local_histos;
    ibooker.setCurrentFolder(subdir + "/" + key);
    edm::LogInfo("Phase2OTMonitorRecHit") << " Booking Histograms in : " << key;

    if (tkGeom_->getDetectorType(det_id) == TrackerGeometry::ModuleType::Ph2PSP) {
      local_histos.numberRecHits_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NRecHitsLayer_P"), ibooker);

      local_histos.clusterSize_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize_P"), ibooker);

      local_histos.globalPosXY_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_perlayer_P"), ibooker);

      local_histos.localPosXY_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY_P"), ibooker);

    }  //if block for P

    ibooker.setCurrentFolder(subdir + "/" + key);
    local_histos.numberRecHits_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NRecHitsLayer_P"), ibooker);
    local_histos.clusterSize_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize_S"), ibooker);
    local_histos.localPosXY_S =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY_S"), ibooker);

    layerMEs_.insert(std::make_pair(key, local_histos));
  }
}

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTMonitorRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // rechitMonitorOT
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfRecHits");
    psd0.add<std::string>("title", ";Number of rechits per event;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 000.0);
    psd0.add<int>("NxBins", 150);
    desc.add<edm::ParameterSetDescription>("GlobalNRecHits", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_RecHitPosition_XY_P");
    psd0.add<std::string>("title", "Global_RecHitPosition_XY_P;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_RecHitPosition_XY_S");
    psd0.add<std::string>("title", "Global_RecHitPosition_XY_S;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_S", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_RecHitPosition_RZ_P");
    psd0.add<std::string>("title", "Global_RecHitPosition_RZ_P;z [mm];r [mm]");
    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_RecHitPosition_RZ_S");
    psd0.add<std::string>("title", "Global_RecHitPosition_RZ_S;z [mm];r [mm]");

    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_S", psd0);
  }
  //Layer wise parameter
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfRecHitsLayerP");
    psd0.add<std::string>("title", ";Number of clusters per event(macro pixel sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 28000.0);
    psd0.add<int>("NxBins", 150);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NRecHitsLayer_P", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfRecHitsLayerS");
    psd0.add<std::string>("title", ";Number of clusters per event(strip sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 28000.0);
    psd0.add<int>("NxBins", 150);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NRecHitsLayer_S", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "ClusterSize_P");
    psd0.add<std::string>("title", ";cluster size(macro pixel sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "ClusterSize_S");
    psd0.add<std::string>("title", ";cluster size(strip sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "GlobalPositionXY_perlayer_P");
    psd0.add<std::string>("title", "GlobalRecHitPositionXY_perlayer_P;x[mm];y[mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "GlobalPositionXY_perlayer_S");
    psd0.add<std::string>("title", "GlobalRecHitPositionXY_perlayer_S;x[mm];y[mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalPositionXY_P");
    psd0.add<std::string>("title", "LocalPositionXY_P;x ;y ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -10.0);
    psd0.add<double>("xmax", 10.0);
    psd0.add<int>("NyBins", 50);
    psd0.add<double>("ymin", -10.0);
    psd0.add<double>("ymax", 10.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalPositionXY_S");
    psd0.add<std::string>("title", "LocalPositionXY_S;x ;y ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -10.0);
    psd0.add<double>("xmax", 10.0);
    psd0.add<int>("NyBins", 50);
    psd0.add<double>("ymin", -10.0);
    psd0.add<double>("ymax", 10.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_S", psd0);
  }
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTRecHit");
  desc.add<bool>("Verbosity", false);
  desc.add<edm::InputTag>("rechitsSrc", edm::InputTag("siPhase2RecHits"));
  descriptions.add("Phase2OTMonitorRecHit", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2OTMonitorRecHit);
