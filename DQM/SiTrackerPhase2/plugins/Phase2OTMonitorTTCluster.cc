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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
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
  MonitorElement *Cluster_IMem_Barrel = nullptr;
  MonitorElement *Cluster_IMem_Endcap_Disc = nullptr;
  MonitorElement *Cluster_IMem_Endcap_Ring = nullptr;
  MonitorElement *Cluster_IMem_Endcap_Ring_Fw[trklet::N_DISK] = {};
  MonitorElement *Cluster_IMem_Endcap_Ring_Bw[trklet::N_DISK] = {};
  MonitorElement *Cluster_OMem_Barrel = nullptr;
  MonitorElement *Cluster_OMem_Endcap_Disc = nullptr;
  MonitorElement *Cluster_OMem_Endcap_Ring = nullptr;
  MonitorElement *Cluster_OMem_Endcap_Ring_Fw[trklet::N_DISK] = {};
  MonitorElement *Cluster_OMem_Endcap_Ring_Bw[trklet::N_DISK] = {};
  MonitorElement *Cluster_W = nullptr;
  MonitorElement *Cluster_Phi = nullptr;
  MonitorElement *Cluster_R = nullptr;
  MonitorElement *Cluster_Eta = nullptr;

  MonitorElement *Cluster_Barrel_XY = nullptr;
  MonitorElement *Cluster_Endcap_Fw_XY = nullptr;
  MonitorElement *Cluster_Endcap_Bw_XY = nullptr;
  MonitorElement *Cluster_RZ = nullptr;

private:
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
          Cluster_IMem_Barrel->Fill(tTopo_->layer(detIdClu));
        else
          Cluster_OMem_Barrel->Fill(tTopo_->layer(detIdClu));

        Cluster_Barrel_XY->Fill(posClu.x(), posClu.y());

      }  // end if isBarrel
      else if (detIdClu.subdetId() == static_cast<int>(StripSubdetector::TID))  // Phase 2 Outer Tracker Endcap
      {
        if (memberClu == 0) {
          Cluster_IMem_Endcap_Disc->Fill(tTopo_->layer(detIdClu));  // returns wheel
          Cluster_IMem_Endcap_Ring->Fill(tTopo_->tidRing(detIdClu));
        } else {
          Cluster_OMem_Endcap_Disc->Fill(tTopo_->layer(detIdClu));  // returns wheel
          Cluster_OMem_Endcap_Ring->Fill(tTopo_->tidRing(detIdClu));
        }

        if (posClu.z() > 0) {
          Cluster_Endcap_Fw_XY->Fill(posClu.x(), posClu.y());
          if (memberClu == 0)
            Cluster_IMem_Endcap_Ring_Fw[tTopo_->layer(detIdClu) - 1]->Fill(tTopo_->tidRing(detIdClu));
          else
            Cluster_OMem_Endcap_Ring_Fw[tTopo_->layer(detIdClu) - 1]->Fill(tTopo_->tidRing(detIdClu));
        } else {
          Cluster_Endcap_Bw_XY->Fill(posClu.x(), posClu.y());
          if (memberClu == 0)
            Cluster_IMem_Endcap_Ring_Bw[tTopo_->layer(detIdClu) - 1]->Fill(tTopo_->tidRing(detIdClu));
          else
            Cluster_OMem_Endcap_Ring_Bw[tTopo_->layer(detIdClu) - 1]->Fill(tTopo_->tidRing(detIdClu));
        }

      }  // end if isEndcap
    }  // end loop contentIter
  }  // end loop inputIter
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

  Cluster_RZ = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_RZ"), iBooker);

  // Barrel Summaries
  iBooker.setCurrentFolder(topFolderName_ + "/Barrel/");
  Cluster_Barrel_XY = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Barrel_XY"), iBooker);
  Cluster_IMem_Barrel = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_IMem_Barrel"), iBooker);
  Cluster_OMem_Barrel = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_OMem_Barrel"), iBooker);

  // BW Endcap Summaries
  iBooker.setCurrentFolder(topFolderName_ + "/EndCaps/MINUS/");
  Cluster_Endcap_Bw_XY = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Endcap_Bw_XY"), iBooker);
  for (int i = 0; i < static_cast<int>(trklet::N_DISK); i++) {
    const std::string si = std::to_string(i + 1);
    Cluster_IMem_Endcap_Ring_Bw[i] =
        book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_IMem_Disc_Bw_" + si), iBooker);
    Cluster_OMem_Endcap_Ring_Bw[i] =
        book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_OMem_Disc_Bw_" + si), iBooker);
  }

  // FW Endcap Summaries
  iBooker.setCurrentFolder(topFolderName_ + "/EndCaps/PLUS/");
  Cluster_Endcap_Fw_XY = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Cluster_Endcap_Fw_XY"), iBooker);
  for (int i = 0; i < static_cast<int>(trklet::N_DISK); i++) {
    const std::string si = std::to_string(i + 1);
    Cluster_IMem_Endcap_Ring_Fw[i] =
        book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_IMem_Disc_Fw_" + si), iBooker);
    Cluster_OMem_Endcap_Ring_Fw[i] =
        book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_OMem_Disc_Fw_" + si), iBooker);
  }

  // Endcap Summaries
  iBooker.setCurrentFolder(topFolderName_ + "/EndCaps/");
  Cluster_IMem_Endcap_Disc =
      book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_IMem_Endcap_Disc"), iBooker);
  Cluster_OMem_Endcap_Disc =
      book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_OMem_Endcap_Disc"), iBooker);
  Cluster_IMem_Endcap_Ring =
      book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_IMem_Endcap_Ring"), iBooker);
  Cluster_OMem_Endcap_Ring =
      book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NL1Clusters_OMem_Endcap_Ring"), iBooker);
}

void Phase2OTMonitorTTCluster::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // NClusters
  phase2tkutil::add1DDesc(
      desc, "NL1Clusters_IMem_Barrel", "NL1Clusters_IMem_Barrel", "Barrel Layer", "# L1 Clusters", 7, 0.5, 7.5);
  phase2tkutil::add1DDesc(
      desc, "NL1Clusters_OMem_Barrel", "NL1Clusters_OMem_Barrel", "Barrel Layer", "# L1 Clusters", 7, 0.5, 7.5);
  phase2tkutil::add1DDesc(
      desc, "NL1Clusters_IMem_Endcap_Disc", "NL1Clusters_IMem_Endcap_Disc", "Endcap Disc", "# L1 Clusters", 6, 0.5, 6.5);
  phase2tkutil::add1DDesc(
      desc, "NL1Clusters_OMem_Endcap_Disc", "NL1Clusters_OMem_Endcap_Disc", "Endcap Disc", "# L1 Clusters", 6, 0.5, 6.5);
  phase2tkutil::add1DDesc(desc,
                          "NL1Clusters_IMem_Endcap_Ring",
                          "NL1Clusters_IMem_Endcap_Ring",
                          "Endcap Ring",
                          "# L1 Clusters",
                          16,
                          0.5,
                          16.5);
  phase2tkutil::add1DDesc(desc,
                          "NL1Clusters_OMem_Endcap_Ring",
                          "NL1Clusters_OMem_Endcap_Ring",
                          "Endcap Ring",
                          "# L1 Clusters",
                          16,
                          0.5,
                          16.5);

  for (int i = 1; i <= static_cast<int>(trklet::N_DISK); i++) {
    const std::string si = std::to_string(i);
    phase2tkutil::add1DDesc(desc,
                            "NL1Clusters_IMem_Disc_Fw_" + si,
                            "NL1Clusters_IMem_Disc+" + si,
                            "Endcap Ring",
                            "# L1 Clusters",
                            16,
                            0.5,
                            16.5);
    phase2tkutil::add1DDesc(desc,
                            "NL1Clusters_IMem_Disc_Bw_" + si,
                            "NL1Clusters_IMem_Disc-" + si,
                            "Endcap Ring",
                            "# L1 Clusters",
                            16,
                            0.5,
                            16.5);
    phase2tkutil::add1DDesc(desc,
                            "NL1Clusters_OMem_Disc_Fw_" + si,
                            "NL1Clusters_OMem_Disc+" + si,
                            "Endcap Ring",
                            "# L1 Clusters",
                            16,
                            0.5,
                            16.5);
    phase2tkutil::add1DDesc(desc,
                            "NL1Clusters_OMem_Disc_Bw_" + si,
                            "NL1Clusters_OMem_Disc-" + si,
                            "Endcap Ring",
                            "# L1 Clusters",
                            16,
                            0.5,
                            16.5);
  }

  // Cluster properties
  phase2tkutil::add2DDesc(
      desc, "L1Cluster_W", "L1Cluster_W", "L1 Cluster Width", "Stack Member", 7, -0.5, 6.5, 2, -0.5, 1.5);
  phase2tkutil::add1DDesc(desc, "L1Cluster_Eta", "L1Cluster_Eta", "#eta", "# L1 Clusters", 45, -5.0, 5.0);
  phase2tkutil::add1DDesc(desc, "L1Cluster_Phi", "L1Cluster_Phi", "#phi", "# L1 Clusters", 60, -3.5, 3.5);
  phase2tkutil::add1DDesc(desc, "L1Cluster_R", "L1Cluster_R", "R [cm]", "# L1 Clusters", 45, 0, 120);

  // Position
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Barrel_XY",
                          "L1Cluster_Barrel_XY",
                          "L1 Cluster Barrel position x [cm]",
                          "L1 Cluster Barrel position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Endcap_Fw_XY",
                          "L1Cluster_Endcap_Fw_XY",
                          "L1 Cluster Forward Endcap position x [cm]",
                          "L1 Cluster Forward Endcap position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_Endcap_Bw_XY",
                          "L1Cluster_Endcap_Bw_XY",
                          "L1 Cluster Backward Endcap position x [cm]",
                          "L1 Cluster Backward Endcap position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Cluster_RZ",
                          "L1Cluster_RZ",
                          "L1 Cluster position z [cm]",
                          "L1 Cluster position #rho [cm]",
                          900,
                          -300,
                          300,
                          900,
                          0,
                          120);

  desc.add<std::string>("TopFolderName", "OuterTracker");
  desc.add<edm::InputTag>("TTClusters", edm::InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"));
  descriptions.add("Phase2OTMonitorTTCluster", desc);
}
DEFINE_FWK_MODULE(Phase2OTMonitorTTCluster);
