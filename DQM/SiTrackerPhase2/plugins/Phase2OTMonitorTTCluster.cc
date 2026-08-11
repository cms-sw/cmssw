// -*- C++ -*-
//
// Package:    SiOuterTracker
// Class:      SiOuterTracker
//
/**\class SiOuterTracker Phase2OTMonitorTTCluster.cc
 DQM/SiOuterTracker/plugins/Phase2OTMonitorTTCluster.cc

 Description: [one line class summary]

 Implementation:
 [Notes on implementation]
 */
//
// Original Author:  Isabelle Helena J De Bruyn
//         Created:  Mon, 10 Feb 2014 13:57:08 GMT
//

// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

class Phase2OTMonitorTTCluster : public DQMEDAnalyzer {
public:
  explicit Phase2OTMonitorTTCluster(const edm::ParameterSet &);
  ~Phase2OTMonitorTTCluster() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  // TTCluster stacks
  MonitorElement *NClusters_Barrel = nullptr;
  MonitorElement *NClusters_IMem_Barrel = nullptr;
  MonitorElement *NClusters_OMem_Barrel = nullptr;

private:
  struct TTClusterMEs {
    MonitorElement *NClusters = nullptr;
    MonitorElement *NClustersIMem = nullptr;
    MonitorElement *NClustersOMem = nullptr;
    MonitorElement *NClustersByRing = nullptr;
    MonitorElement *NClustersIMemByRing = nullptr;
    MonitorElement *NClustersOMemByRing = nullptr;
    MonitorElement *NClustersByWheel = nullptr;
    MonitorElement *NClustersIMemByWheel = nullptr;
    MonitorElement *NClustersOMemByWheel = nullptr;
    unsigned int clusterCounter = 0;
    unsigned int clusterCounterIMem = 0;
    unsigned int clusterCounterOMem = 0;
  };

  MonitorElement *Cluster_W = nullptr;
  MonitorElement *Cluster_Phi = nullptr;
  MonitorElement *Cluster_R = nullptr;
  MonitorElement *Cluster_Eta = nullptr;

  MonitorElement *Cluster_Barrel_XY = nullptr;
  MonitorElement *Cluster_Endcap_Fw_XY = nullptr;
  MonitorElement *Cluster_Endcap_Bw_XY = nullptr;
  MonitorElement *Cluster_RZ = nullptr;

  void bookLayerHistos(DQMStore::IBooker &ibooker, uint32_t det_id, std::string &subdir);

  std::map<std::string, TTClusterMEs> layerMEs_;
  enum Level { OT = 1, SUBSTRUCTURE, ENDCAP_SIDE, ENDCAP_RING, ENDCAP_WHEEL, LAYER };

  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> tagTTClustersToken_;
  std::string topFolderName_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry *tkGeom_ = nullptr;
  const TrackerTopology *tTopo_ = nullptr;
};

//
// constructors and destructor
//
Phase2OTMonitorTTCluster::Phase2OTMonitorTTCluster(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTClustersToken_ = consumes<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>(
      conf_.getParameter<edm::InputTag>("TTClusters"));
}

Phase2OTMonitorTTCluster::~Phase2OTMonitorTTCluster() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
void Phase2OTMonitorTTCluster::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  tkGeom_ = &(iSetup.getData(geomToken_));
  tTopo_ = &(iSetup.getData(topoToken_));
}

// ------------ method called for each event  ------------
void Phase2OTMonitorTTCluster::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  /// Track Trigger Clusters
  edm::Handle<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTClusterHandle;
  iEvent.getByToken(tagTTClustersToken_, Phase2TrackerDigiTTClusterHandle);

  /// Loop over the input Clusters
  typename edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
  typename edmNew::DetSet<TTCluster<Ref_Phase2TrackerDigi_>>::const_iterator contentIter;

  // Adding protection
  if (!Phase2TrackerDigiTTClusterHandle.isValid())
    return;

  for (inputIter = Phase2TrackerDigiTTClusterHandle->begin(); inputIter != Phase2TrackerDigiTTClusterHandle->end();
       ++inputIter) {
    for (contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter) {
      // Make reference cluster
      edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>, TTCluster<Ref_Phase2TrackerDigi_>> tempCluRef =
          edmNew::makeRefTo(Phase2TrackerDigiTTClusterHandle, contentIter);

      DetId detIdClu = tkGeom_->idToDet(tempCluRef->getDetId())->geographicalId();
      unsigned int memberClu = tempCluRef->getStackMember();
      unsigned int widClu = tempCluRef->findWidth();

      MeasurementPoint mp = tempCluRef->findAverageLocalCoordinates();
      const GeomDet *theGeomDet = tkGeom_->idToDet(detIdClu);
      Global3DPoint posClu = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(mp));

      double r = posClu.perp();
      double z = posClu.z();

      Cluster_W->Fill(widClu, memberClu);
      Cluster_Eta->Fill(posClu.eta());
      Cluster_Phi->Fill(posClu.phi());
      Cluster_R->Fill(r);
      Cluster_RZ->Fill(z, r);

      if (detIdClu.subdetId() == static_cast<int>(StripSubdetector::TOB))  // Phase 2 Outer Tracker Barrel
      {
        if (memberClu == 0)
          NClusters_IMem_Barrel->Fill(tTopo_->layer(detIdClu));
        else
          NClusters_OMem_Barrel->Fill(tTopo_->layer(detIdClu));

        NClusters_Barrel->Fill(tTopo_->layer(detIdClu));
        Cluster_Barrel_XY->Fill(posClu.x(), posClu.y());

      }  // end if isBarrel
      for (enum Level fillingDepth = OT; fillingDepth <= LAYER; fillingDepth = Level(fillingDepth + 1)) {
        // Skip filling for barrel detIds on endcap-only depths
        if ((fillingDepth >= ENDCAP_SIDE && fillingDepth < LAYER) &&
            DetId(detIdClu).subdetId() == SiStripSubdetector::TOB)
          continue;
        std::string folderKey = phase2tkutil::getHistoId(detIdClu, tTopo_, 0, fillingDepth, false);
        auto layerMEiter = layerMEs_.find(folderKey);
        if (layerMEiter == layerMEs_.end())
          continue;
        TTClusterMEs &local_mes = layerMEiter->second;

        local_mes.clusterCounter++;
        if (memberClu == 0)
          local_mes.clusterCounterIMem++;
        else
          local_mes.clusterCounterOMem++;

        if (detIdClu.subdetId() == static_cast<int>(StripSubdetector::TID)) {
          if (local_mes.NClustersByWheel)
            local_mes.NClustersByWheel->Fill(tTopo_->tidWheel(detIdClu));
          if (local_mes.NClustersByRing)
            local_mes.NClustersByRing->Fill(tTopo_->tidRing(detIdClu));
          if (memberClu == 0) {
            if (local_mes.NClustersIMemByWheel)
              local_mes.NClustersIMemByWheel->Fill(tTopo_->tidWheel(detIdClu));
            if (local_mes.NClustersIMemByRing)
              local_mes.NClustersIMemByRing->Fill(tTopo_->tidRing(detIdClu));
          } else {
            if (local_mes.NClustersOMemByWheel)
              local_mes.NClustersOMemByWheel->Fill(tTopo_->tidWheel(detIdClu));
            if (local_mes.NClustersOMemByRing)
              local_mes.NClustersOMemByRing->Fill(tTopo_->tidRing(detIdClu));
          }
        }
      }  // end loop fillingDepth
    }  // end loop contentIter
  }  // end loop inputIter
  for (const auto &it : layerMEs_) {
    TTClusterMEs local_mes = it.second;
    if (local_mes.NClusters)
      local_mes.NClusters->Fill(local_mes.clusterCounter);
    local_mes.clusterCounter = 0;
    if (local_mes.NClustersIMem)
      local_mes.NClustersIMem->Fill(local_mes.clusterCounterIMem);
    local_mes.clusterCounterIMem = 0;
    if (local_mes.NClustersOMem)
      local_mes.NClustersOMem->Fill(local_mes.clusterCounterOMem);
    local_mes.clusterCounterOMem = 0;
  }
}  // end of method

// ------------ method called once each job just before starting event loop
// ------------
void Phase2OTMonitorTTCluster::bookHistograms(DQMStore::IBooker &iBooker,
                                              edm::Run const &run,
                                              edm::EventSetup const &es) {
  using namespace phase2tkutil;

  // Whole OT Summaries
  iBooker.setCurrentFolder(topFolderName_);
  Cluster_W = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_W"), iBooker);
  Cluster_Eta = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Eta"), iBooker);
  Cluster_Phi = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Phi"), iBooker);
  Cluster_R = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_R"), iBooker);

  // Positions
  iBooker.setCurrentFolder(topFolderName_ + "/Positions/");
  Cluster_RZ = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Global_Position_RZ"), iBooker);
  Cluster_Barrel_XY =
      book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Global_Position_Barrel_XY"), iBooker);
  Cluster_Endcap_Bw_XY =
      book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Global_Position_Endcap_Bw_XY"), iBooker);
  Cluster_Endcap_Fw_XY =
      book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Global_Position_Endcap_Fw_XY"), iBooker);

  // Barrel Summaries
  iBooker.setCurrentFolder(topFolderName_ + "/Barrel/");
  NClusters_Barrel = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Num_L1Clusters_Barrel"), iBooker);
  NClusters_IMem_Barrel = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Num_L1Clusters_IMem_Barrel"), iBooker);
  NClusters_OMem_Barrel = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Num_L1Clusters_OMem_Barrel"), iBooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(es)) {
    for (auto const &det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      //continue if Pixel
      if ((det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
           det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      edm::LogInfo("Phase2OTMonitorTTCluster")
          << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
          << "\t key=" << phase2tkutil::getHistoId(detId_raw, tTopo_, 0.0, 6, false) << std::endl;
      bookLayerHistos(iBooker, detId_raw, topFolderName_);
    }
  }
}

//////////////////Layer Histo/////////////////////////////////
void Phase2OTMonitorTTCluster::bookLayerHistos(DQMStore::IBooker &ibooker, uint32_t det_id, std::string &subdir) {
  for (enum Level bookingDepth = OT; bookingDepth <= LAYER; bookingDepth = Level(bookingDepth + 1)) {
    std::string folderName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, true);

    if (layerMEs_.find(folderName) == layerMEs_.end()) {
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + folderName);
      edm::LogInfo("Phase2OTMonitorTTCluster") << " Booking Histograms in: " << subdir + "/" + folderName;
      TTClusterMEs local_mes;

      // If this det is a barrel det AND bookingDepth is an endcap-only depth, DO NOT BOOK
      if ((bookingDepth >= ENDCAP_SIDE && bookingDepth < LAYER) &&
          DetId(det_id).subdetId() == static_cast<int>(StripSubdetector::TOB))
        continue;

      local_mes.NClusters = phase2tkutil::book1DFromPSet(
          conf_.getParameter<edm::ParameterSet>("NClustersLayer"), ibooker, prettyName, bookingDepth);
      local_mes.NClustersIMem = phase2tkutil::book1DFromPSet(
          conf_.getParameter<edm::ParameterSet>("NClustersIMemLayer"), ibooker, prettyName, bookingDepth);
      local_mes.NClustersOMem = phase2tkutil::book1DFromPSet(
          conf_.getParameter<edm::ParameterSet>("NClustersOMemLayer"), ibooker, prettyName, bookingDepth);

      if (DetId(det_id).subdetId() == static_cast<int>(StripSubdetector::TID)) {
        if (bookingDepth >= SUBSTRUCTURE && bookingDepth < LAYER) {
          if (bookingDepth != ENDCAP_WHEEL) {
            local_mes.NClustersByWheel = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NClustersByWheel"), ibooker, prettyName);
            local_mes.NClustersIMemByWheel = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NClustersIMemByWheel"), ibooker, prettyName);
            local_mes.NClustersOMemByWheel = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NClustersOMemByWheel"), ibooker, prettyName);
          }
          if (bookingDepth != ENDCAP_RING) {
            local_mes.NClustersByRing = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NClustersByRing"), ibooker, prettyName);
            local_mes.NClustersIMemByRing = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NClustersIMemByRing"), ibooker, prettyName);
            local_mes.NClustersOMemByRing = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NClustersOMemByRing"), ibooker, prettyName);
          }
        }
      }
      layerMEs_.emplace(folderName, local_mes);
    }
  }
}

void Phase2OTMonitorTTCluster::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // NClusters
  phase2tkutil::add1DDesc(desc,
                          "Num_L1Clusters_IMem_Barrel",
                          "Num_L1Clusters_IMem_Barrel_Layers",
                          "Number of L1Clusters in inner member of modules in the barrel by layer",
                          "Barrel Layer",
                          "# L1 Clusters",
                          7,
                          0.5,
                          7.5);
  phase2tkutil::add1DDesc(desc,
                          "Num_L1Clusters_OMem_Barrel",
                          "Num_L1Clusters_OMem_Barrel_Layers",
                          "Number of L1Clusters in outer member of modules in the barrel by layer",
                          "Barrel Layer",
                          "# L1 Clusters",
                          7,
                          0.5,
                          7.5);
  phase2tkutil::add1DDesc(desc,
                          "Num_L1Clusters_Barrel",
                          "Num_L1Clusters_Barrel_Layers",
                          "Number of L1Clusters in the barrel by layer",
                          "Barrel Layer",
                          "# L1 Clusters",
                          7,
                          0.5,
                          7.5);

  // Cluster properties
  phase2tkutil::add2DDesc(
      desc, "L1Cluster_W", "L1Cluster_W", "L1Cluster_W", "L1 Cluster Width", "Stack Member", 7, -0.5, 6.5, 2, -0.5, 1.5);
  phase2tkutil::add1DDesc(
      desc, "L1Cluster_Eta", "L1Cluster_Eta", "L1Cluster_Eta", "#eta", "# L1 Clusters", 45, -5.0, 5.0);
  phase2tkutil::add1DDesc(
      desc, "L1Cluster_Phi", "L1Cluster_Phi", "L1Cluster_Phi", "#phi", "# L1 Clusters", 60, -3.5, 3.5);
  phase2tkutil::add1DDesc(desc, "L1Cluster_R", "L1Cluster_R", "L1Cluster_R", "R [cm]", "# L1 Clusters", 45, 0, 120);

  // Position
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Global_Position_Barrel_XY",
                          "L1Cluster_Global_Position_Barrel_XY",
                          "L1Cluster_Global_Position_Barrel_XY",
                          "L1 Cluster Barrel position x [cm]",
                          "L1 Cluster Barrel position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Global_Position_Endcap_Fw_XY",
                          "L1Cluster_Global_Position_Endcap_Fw_XY",
                          "L1Cluster_Global_Position_Endcap_Fw_XY",
                          "L1 Cluster Forward Endcap position x [cm]",
                          "L1 Cluster Forward Endcap position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Global_Position_Endcap_Bw_XY",
                          "L1Cluster_Global_Position_Endcap_Bw_XY",
                          "L1Cluster_Global_Position_Endcap_Bw_XY",
                          "L1 Cluster Backward Endcap position x [cm]",
                          "L1 Cluster Backward Endcap position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Global_Position_RZ",
                          "L1Cluster_Global_Position_RZ",
                          "L1Cluster_Global_Position_RZ",
                          "L1 Cluster position z [cm]",
                          "L1 Cluster position #rho [cm]",
                          900,
                          -300,
                          300,
                          900,
                          0,
                          120);

  // Layer-wise params
  // counts
  phase2tkutil::add1DDesc(desc,
                          "NClustersLayer",
                          "Num_L1Clusters",
                          "Number of L1Clusters in {} per event",
                          "Number of clusters",
                          "Number of events",
                          100,
                          0,
                          300000);
  phase2tkutil::add1DDesc(desc,
                          "NClustersIMemLayer",
                          "Num_L1Clusters_IMem",
                          "Number of L1Clusters in inner member of modules in {} per event",
                          "Number of clusters",
                          "Number of events",
                          100,
                          0,
                          300000);
  phase2tkutil::add1DDesc(desc,
                          "NClustersOMemLayer",
                          "Num_L1Clusters_OMem",
                          "Number of L1Clusters in outer member of modules in {} per event",
                          "Number of clusters",
                          "Number of events",
                          100,
                          0,
                          300000);

  // endcap
  phase2tkutil::add1DDesc(desc,
                          "NClustersByWheel",
                          "Num_L1Clusters_Wheels",
                          "Number of L1Clusters in {} by wheel",
                          "Wheel",
                          "Number of clusters",
                          6,
                          0.5,
                          6.5);
  phase2tkutil::add1DDesc(desc,
                          "NClustersIMemByWheel",
                          "Num_L1Clusters_Wheels_IMem",
                          "Number of L1Clusters in inner member of modules in {} by wheel",
                          "Wheel",
                          "Number of clusters",
                          6,
                          0.5,
                          6.5);
  phase2tkutil::add1DDesc(desc,
                          "NClustersOMemByWheel",
                          "Num_L1Clusters_Wheels_OMem",
                          "Number of L1Clusters in inner member of modules in {} by wheel",
                          "Wheel",
                          "Number of clusters",
                          6,
                          0.5,
                          6.5);

  //wheel
  phase2tkutil::add1DDesc(desc,
                          "NClustersByRing",
                          "Num_L1Clusters_Rings",
                          "Number of L1Clusters in {} by ring",
                          "Ring",
                          "Number of clusters",
                          16,
                          0.5,
                          16.5);
  phase2tkutil::add1DDesc(desc,
                          "NClustersIMemByRing",
                          "Num_L1Clusters_Rings_IMem",
                          "Number of L1Clusters in inner member of modules in {} by ring",
                          "Ring",
                          "Number of clusters",
                          16,
                          0.5,
                          16.5);
  phase2tkutil::add1DDesc(desc,
                          "NClustersOMemByRing",
                          "Num_L1Clusters_Rings_OMem",
                          "Number of L1Clusters in outer member of modules in {} by ring",
                          "Ring",
                          "Number of clusters",
                          16,
                          0.5,
                          16.5);

  desc.add<std::string>("TopFolderName", "OuterTracker");
  desc.add<edm::InputTag>("TTClusters", edm::InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"));
  descriptions.add("Phase2OTMonitorTTCluster", desc);
}
DEFINE_FWK_MODULE(Phase2OTMonitorTTCluster);
