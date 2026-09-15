// Package:    Phase2OTMonitorRecHit
// Class:      Phase2OTMonitorRecHit
//
/**\class Phase2OTMonitorRecHit Phase2OTMonitorRecHit.cc 
 Description:  Standalone  Plugin for Phase2 RecHit validation
*/
//
// Author: Suvankar Roy Chowdhury
// Date: March 2021
// Date: August 2026 (modified by Lisa Juckett for dqm output folder restructure)
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
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/TrackerGeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"

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

  MonitorElement* globalXY_P_ = nullptr;
  MonitorElement* globalRZ_P_ = nullptr;
  MonitorElement* globalXY_S_ = nullptr;
  MonitorElement* globalRZ_S_ = nullptr;
  struct RecHitME {
    // use TH1D instead of TH1F to avoid stauration at 2^31
    // above this increments with +1 don't work for float, need double
    MonitorElement* numberRecHits_P = nullptr;
    MonitorElement* numberRecHits_S = nullptr;
    MonitorElement* clusterSize_P = nullptr;
    MonitorElement* clusterSize_S = nullptr;
    unsigned int recHitCounter_P = 0;
    unsigned int recHitCounter_S = 0;
  };
  std::map<std::string, RecHitME> layerMEs_;
  enum Level { OT = 1, SUBSTRUCTURE, ENDCAP_SIDE, ENDCAP_RING, ENDCAP_WHEEL, LAYER };
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
}

//
// -- Analyze
//
void Phase2OTMonitorRecHit::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the RecHits Phase2TrackerRecHit1DCollectionNew
  const auto& rechits = iEvent.getHandle(tokenRecHitsOT_);
  if (!rechits.isValid())
    return;
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
      } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
        globalXY_S_->Fill(gx, gy);
        globalRZ_S_->Fill(gz, gr);
      }
      for (enum Level fillingDepth = OT; fillingDepth <= LAYER; fillingDepth = Level(fillingDepth + 1)) {
        // Skip filling for barrel detIds on endcap-only depths
        if ((fillingDepth >= ENDCAP_SIDE && fillingDepth < LAYER) && DetId(detId).subdetId() == SiStripSubdetector::TOB)
          continue;
        std::string key = phase2tkutil::getHistoId(detId, tTopo_, 0.0, fillingDepth, false);

        if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
          if (layerMEs_[key].clusterSize_P)
            layerMEs_[key].clusterSize_P->Fill(rechitIt->cluster()->size());
          layerMEs_[key].recHitCounter_P++;
        } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
          if (layerMEs_[key].clusterSize_S)
            layerMEs_[key].clusterSize_S->Fill(rechitIt->cluster()->size());
          layerMEs_[key].recHitCounter_S++;
        }
      }
    }  //end loop over rechits of a detId
  }  //End loop over DetSetVector

  //fill nRecHit counter per layer
  for (auto& lme : layerMEs_) {
    RecHitME& local_mes = lme.second;
    if (local_mes.numberRecHits_P)
      local_mes.numberRecHits_P->Fill(local_mes.recHitCounter_P);
    local_mes.recHitCounter_P = 0;
    if (local_mes.numberRecHits_S)
      local_mes.numberRecHits_S->Fill(local_mes.recHitCounter_S);
    local_mes.recHitCounter_S = 0;
  }
}
//
// -- Book Histograms
//
void Phase2OTMonitorRecHit::bookHistograms(DQMStore::IBooker& ibooker,
                                           edm::Run const& iRun,
                                           edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");

  ibooker.cd();
  edm::LogInfo("Phase2OTMonitorRecHit") << " Booking Histograms in : " << top_folder;
  ibooker.setCurrentFolder(top_folder);

  //Global histos for OT
  ibooker.setCurrentFolder(top_folder + "/Positions/");

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
  for (enum Level bookingDepth = OT; bookingDepth <= LAYER; bookingDepth = Level(bookingDepth + 1)) {
    // If this det is a barrel det AND bookingDepth is an endcap-only depth, DO NOT BOOK
    if ((bookingDepth >= ENDCAP_SIDE && bookingDepth < LAYER) && DetId(det_id).subdetId() == SiStripSubdetector::TOB)
      continue;
    std::string key = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, true);

    if (layerMEs_.find(key) == layerMEs_.end()) {
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + key);

      RecHitME local_histos;
      edm::LogInfo("Phase2OTMonitorRecHit") << " Booking Histograms in : " << key;

      if (tkGeom_->getDetectorType(det_id) == TrackerGeometry::ModuleType::Ph2PSP) {
        local_histos.numberRecHits_P = phase2tkutil::book1DFromPSet(
            config_.getParameter<edm::ParameterSet>("NRecHitsLayer_P"), ibooker, prettyName, bookingDepth);
        local_histos.clusterSize_P =
            phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("RecHitSize_P"), ibooker, prettyName);
      }  //if block for P

      local_histos.numberRecHits_S = phase2tkutil::book1DFromPSet(
          config_.getParameter<edm::ParameterSet>("NRecHitsLayer_S"), ibooker, prettyName, bookingDepth);
      local_histos.clusterSize_S =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("RecHitSize_S"), ibooker, prettyName);

      layerMEs_.insert(std::make_pair(key, local_histos));
    }
  }
}

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTMonitorRecHit::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_P",
                          "RecHit_Global_Position_XY_P",
                          "Global_RecHitPosition_XY_P",
                          "x [mm]",
                          "y [mm]",
                          1250,
                          -1250.0,
                          1250.0,
                          1250,
                          -1250.0,
                          1250.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_S",
                          "RecHit_Global_Position_XY_S",
                          "Global_RecHitPosition_XY_S",
                          "x [mm]",
                          "y [mm]",
                          1250,
                          -1250.0,
                          1250.0,
                          1250,
                          -1250.0,
                          1250.0);

  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_P",
                          "RecHit_Global_Position_RZ_P",
                          "Global_RecHitPosition_RZ_P",
                          "z [mm]",
                          "r [mm]",
                          1500,
                          -3000.0,
                          3000.0,
                          1250,
                          0.0,
                          1250.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_S",
                          "RecHit_Global_Position_RZ_S",
                          "Global_RecHitPosition_RZ_S",
                          "z [mm]",
                          "r [mm]",
                          1500,
                          -3000.0,
                          3000.0,
                          1250,
                          0.0,
                          1250.0);

  //Layer wise parameter
  phase2tkutil::add1DDesc(desc,
                          "NRecHitsLayer_P",
                          "Num_RecHits_Per_Event_P",
                          "Number of RecHits per event in pixels in {}",
                          "Number of rechits",
                          "Number of events",
                          150,
                          0.0,
                          300000.0);
  phase2tkutil::add1DDesc(desc,
                          "NRecHitsLayer_S",
                          "Num_RecHits_Per_Event_S",
                          "Number of RecHits per event in strips in {}",
                          "Number of rechits",
                          "Number of events",
                          150,
                          0.0,
                          300000.0);

  phase2tkutil::add1DDesc(desc,
                          "RecHitSize_P",
                          "RecHit_Size_P",
                          "RecHit size in pixels in {}",
                          "RecHit size (macro pixel)",
                          "",
                          31,
                          -0.5,
                          30.5);
  phase2tkutil::add1DDesc(
      desc, "RecHitSize_S", "RecHit_Size_S", "RecHit size in strips in {}", "RecHit size (strips)", "", 31, -0.5, 30.5);

  desc.add<std::string>("TopFolderName", "OuterTracker");
  desc.add<bool>("Verbosity", false);
  desc.add<edm::InputTag>("rechitsSrc", edm::InputTag("siPhase2RecHits"));
  descriptions.add("Phase2OTMonitorRecHit", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2OTMonitorRecHit);
