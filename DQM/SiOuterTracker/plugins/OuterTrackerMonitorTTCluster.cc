// -*- C++ -*-
//
// Package:    SiOuterTracker
// Class:      SiOuterTracker
//
/**\class SiOuterTracker OuterTrackerMonitorTTCluster.cc
 DQM/SiOuterTracker/plugins/OuterTrackerMonitorTTCluster.cc

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

class OuterTrackerMonitorTTCluster : public DQMEDAnalyzer {
public:
  explicit OuterTrackerMonitorTTCluster(const edm::ParameterSet &);
  ~OuterTrackerMonitorTTCluster() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  // TTCluster stacks
  MonitorElement *Cluster_IMem_Barrel = nullptr;
  MonitorElement *Cluster_IMem_Endcap_Disc = nullptr;
  MonitorElement *Cluster_IMem_Endcap_Ring = nullptr;
  MonitorElement *Cluster_IMem_Endcap_Ring_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement *Cluster_IMem_Endcap_Ring_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement *Cluster_OMem_Barrel = nullptr;
  MonitorElement *Cluster_OMem_Endcap_Disc = nullptr;
  MonitorElement *Cluster_OMem_Endcap_Ring = nullptr;
  MonitorElement *Cluster_OMem_Endcap_Ring_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement *Cluster_OMem_Endcap_Ring_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
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
OuterTrackerMonitorTTCluster::OuterTrackerMonitorTTCluster(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTClustersToken_ = consumes<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_>>>(
      conf_.getParameter<edm::InputTag>("TTClusters"));
}

OuterTrackerMonitorTTCluster::~OuterTrackerMonitorTTCluster() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
void OuterTrackerMonitorTTCluster::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  tkGeom_ = &(iSetup.getData(geomToken_));
  tTopo_ = &(iSetup.getData(topoToken_));
}

// ------------ method called for each event  ------------
void OuterTrackerMonitorTTCluster::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
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

      }                                                                         // end if isBarrel
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
    }    // end loop contentIter
  }      // end loop inputIter
}  // end of method

// ------------ method called once each job just before starting event loop
// ------------
void OuterTrackerMonitorTTCluster::bookHistograms(DQMStore::IBooker &iBooker,
                                                  edm::Run const &run,
                                                  edm::EventSetup const &es) {
  std::string HistoName;
  const int numDiscs = 5;

  iBooker.setCurrentFolder(topFolderName_ + "/Clusters/NClusters");

  // NClusters
  edm::ParameterSet psTTCluster_Barrel = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Barrel");
  HistoName = "NClusters_IMem_Barrel";
  Cluster_IMem_Barrel = iBooker.book1D(HistoName,
                                       HistoName,
                                       psTTCluster_Barrel.getParameter<int32_t>("Nbinsx"),
                                       psTTCluster_Barrel.getParameter<double>("xmin"),
                                       psTTCluster_Barrel.getParameter<double>("xmax"));
  Cluster_IMem_Barrel->setAxisTitle("Barrel Layer", 1);
  Cluster_IMem_Barrel->setAxisTitle("# L1 Clusters", 2);

  HistoName = "NClusters_OMem_Barrel";
  Cluster_OMem_Barrel = iBooker.book1D(HistoName,
                                       HistoName,
                                       psTTCluster_Barrel.getParameter<int32_t>("Nbinsx"),
                                       psTTCluster_Barrel.getParameter<double>("xmin"),
                                       psTTCluster_Barrel.getParameter<double>("xmax"));
  Cluster_OMem_Barrel->setAxisTitle("Barrel Layer", 1);
  Cluster_OMem_Barrel->setAxisTitle("# L1 Clusters", 2);

  edm::ParameterSet psTTCluster_ECDisc = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_ECDiscs");
  HistoName = "NClusters_IMem_Endcap_Disc";
  Cluster_IMem_Endcap_Disc = iBooker.book1D(HistoName,
                                            HistoName,
                                            psTTCluster_ECDisc.getParameter<int32_t>("Nbinsx"),
                                            psTTCluster_ECDisc.getParameter<double>("xmin"),
                                            psTTCluster_ECDisc.getParameter<double>("xmax"));
  Cluster_IMem_Endcap_Disc->setAxisTitle("Endcap Disc", 1);
  Cluster_IMem_Endcap_Disc->setAxisTitle("# L1 Clusters", 2);

  HistoName = "NClusters_OMem_Endcap_Disc";
  Cluster_OMem_Endcap_Disc = iBooker.book1D(HistoName,
                                            HistoName,
                                            psTTCluster_ECDisc.getParameter<int32_t>("Nbinsx"),
                                            psTTCluster_ECDisc.getParameter<double>("xmin"),
                                            psTTCluster_ECDisc.getParameter<double>("xmax"));
  Cluster_OMem_Endcap_Disc->setAxisTitle("Endcap Disc", 1);
  Cluster_OMem_Endcap_Disc->setAxisTitle("# L1 Clusters", 2);

  edm::ParameterSet psTTCluster_ECRing = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_ECRings");
  HistoName = "NClusters_IMem_Endcap_Ring";
  Cluster_IMem_Endcap_Ring = iBooker.book1D(HistoName,
                                            HistoName,
                                            psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
                                            psTTCluster_ECRing.getParameter<double>("xmin"),
                                            psTTCluster_ECRing.getParameter<double>("xmax"));
  Cluster_IMem_Endcap_Ring->setAxisTitle("Endcap Ring", 1);
  Cluster_IMem_Endcap_Ring->setAxisTitle("# L1 Clusters", 2);

  HistoName = "NClusters_OMem_Endcap_Ring";
  Cluster_OMem_Endcap_Ring = iBooker.book1D(HistoName,
                                            HistoName,
                                            psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
                                            psTTCluster_ECRing.getParameter<double>("xmin"),
                                            psTTCluster_ECRing.getParameter<double>("xmax"));
  Cluster_OMem_Endcap_Ring->setAxisTitle("Endcap Ring", 1);
  Cluster_OMem_Endcap_Ring->setAxisTitle("# L1 Clusters", 2);

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "NClusters_IMem_Disc+" + std::to_string(i + 1);
    Cluster_IMem_Endcap_Ring_Fw[i] = iBooker.book1D(HistoName,
                                                    HistoName,
                                                    psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
                                                    psTTCluster_ECRing.getParameter<double>("xmin"),
                                                    psTTCluster_ECRing.getParameter<double>("xmax"));
    Cluster_IMem_Endcap_Ring_Fw[i]->setAxisTitle("Endcap Ring", 1);
    Cluster_IMem_Endcap_Ring_Fw[i]->setAxisTitle("# L1 Clusters ", 2);
  }

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "NClusters_IMem_Disc-" + std::to_string(i + 1);
    Cluster_IMem_Endcap_Ring_Bw[i] = iBooker.book1D(HistoName,
                                                    HistoName,
                                                    psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
                                                    psTTCluster_ECRing.getParameter<double>("xmin"),
                                                    psTTCluster_ECRing.getParameter<double>("xmax"));
    Cluster_IMem_Endcap_Ring_Bw[i]->setAxisTitle("Endcap Ring", 1);
    Cluster_IMem_Endcap_Ring_Bw[i]->setAxisTitle("# L1 Clusters ", 2);
  }

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "NClusters_OMem_Disc+" + std::to_string(i + 1);
    Cluster_OMem_Endcap_Ring_Fw[i] = iBooker.book1D(HistoName,
                                                    HistoName,
                                                    psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
                                                    psTTCluster_ECRing.getParameter<double>("xmin"),
                                                    psTTCluster_ECRing.getParameter<double>("xmax"));
    Cluster_OMem_Endcap_Ring_Fw[i]->setAxisTitle("Endcap Ring", 1);
    Cluster_OMem_Endcap_Ring_Fw[i]->setAxisTitle("# L1 Clusters ", 2);
  }

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "NClusters_OMem_Disc-" + std::to_string(i + 1);
    Cluster_OMem_Endcap_Ring_Bw[i] = iBooker.book1D(HistoName,
                                                    HistoName,
                                                    psTTCluster_ECRing.getParameter<int32_t>("Nbinsx"),
                                                    psTTCluster_ECRing.getParameter<double>("xmin"),
                                                    psTTCluster_ECRing.getParameter<double>("xmax"));
    Cluster_OMem_Endcap_Ring_Bw[i]->setAxisTitle("Endcap Ring", 1);
    Cluster_OMem_Endcap_Ring_Bw[i]->setAxisTitle("# L1 Clusters ", 2);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Clusters");

  // Cluster Width
  edm::ParameterSet psTTClusterWidth = conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Width");
  HistoName = "Cluster_W";
  Cluster_W = iBooker.book2D(HistoName,
                             HistoName,
                             psTTClusterWidth.getParameter<int32_t>("Nbinsx"),
                             psTTClusterWidth.getParameter<double>("xmin"),
                             psTTClusterWidth.getParameter<double>("xmax"),
                             psTTClusterWidth.getParameter<int32_t>("Nbinsy"),
                             psTTClusterWidth.getParameter<double>("ymin"),
                             psTTClusterWidth.getParameter<double>("ymax"));
  Cluster_W->setAxisTitle("L1 Cluster Width", 1);
  Cluster_W->setAxisTitle("Stack Member", 2);

  // Cluster eta distribution
  edm::ParameterSet psTTClusterEta = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Eta");
  HistoName = "Cluster_Eta";
  Cluster_Eta = iBooker.book1D(HistoName,
                               HistoName,
                               psTTClusterEta.getParameter<int32_t>("Nbinsx"),
                               psTTClusterEta.getParameter<double>("xmin"),
                               psTTClusterEta.getParameter<double>("xmax"));
  Cluster_Eta->setAxisTitle("#eta", 1);
  Cluster_Eta->setAxisTitle("# L1 Clusters ", 2);

  // Cluster phi distribution
  edm::ParameterSet psTTClusterPhi = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Phi");
  HistoName = "Cluster_Phi";
  Cluster_Phi = iBooker.book1D(HistoName,
                               HistoName,
                               psTTClusterPhi.getParameter<int32_t>("Nbinsx"),
                               psTTClusterPhi.getParameter<double>("xmin"),
                               psTTClusterPhi.getParameter<double>("xmax"));
  Cluster_Phi->setAxisTitle("#phi", 1);
  Cluster_Phi->setAxisTitle("# L1 Clusters", 2);

  // Cluster R distribution
  edm::ParameterSet psTTClusterR = conf_.getParameter<edm::ParameterSet>("TH1TTCluster_R");
  HistoName = "Cluster_R";
  Cluster_R = iBooker.book1D(HistoName,
                             HistoName,
                             psTTClusterR.getParameter<int32_t>("Nbinsx"),
                             psTTClusterR.getParameter<double>("xmin"),
                             psTTClusterR.getParameter<double>("xmax"));
  Cluster_R->setAxisTitle("R [cm]", 1);
  Cluster_R->setAxisTitle("# L1 Clusters", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Clusters/Position");

  // Position plots
  edm::ParameterSet psTTCluster_Barrel_XY = conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Position");
  HistoName = "Cluster_Barrel_XY";
  Cluster_Barrel_XY = iBooker.book2D(HistoName,
                                     HistoName,
                                     psTTCluster_Barrel_XY.getParameter<int32_t>("Nbinsx"),
                                     psTTCluster_Barrel_XY.getParameter<double>("xmin"),
                                     psTTCluster_Barrel_XY.getParameter<double>("xmax"),
                                     psTTCluster_Barrel_XY.getParameter<int32_t>("Nbinsy"),
                                     psTTCluster_Barrel_XY.getParameter<double>("ymin"),
                                     psTTCluster_Barrel_XY.getParameter<double>("ymax"));
  Cluster_Barrel_XY->setAxisTitle("L1 Cluster Barrel position x [cm]", 1);
  Cluster_Barrel_XY->setAxisTitle("L1 Cluster Barrel position y [cm]", 2);

  edm::ParameterSet psTTCluster_Endcap_Fw_XY = conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Position");
  HistoName = "Cluster_Endcap_Fw_XY";
  Cluster_Endcap_Fw_XY = iBooker.book2D(HistoName,
                                        HistoName,
                                        psTTCluster_Endcap_Fw_XY.getParameter<int32_t>("Nbinsx"),
                                        psTTCluster_Endcap_Fw_XY.getParameter<double>("xmin"),
                                        psTTCluster_Endcap_Fw_XY.getParameter<double>("xmax"),
                                        psTTCluster_Endcap_Fw_XY.getParameter<int32_t>("Nbinsy"),
                                        psTTCluster_Endcap_Fw_XY.getParameter<double>("ymin"),
                                        psTTCluster_Endcap_Fw_XY.getParameter<double>("ymax"));
  Cluster_Endcap_Fw_XY->setAxisTitle("L1 Cluster Forward Endcap position x [cm]", 1);
  Cluster_Endcap_Fw_XY->setAxisTitle("L1 Cluster Forward Endcap position y [cm]", 2);

  edm::ParameterSet psTTCluster_Endcap_Bw_XY = conf_.getParameter<edm::ParameterSet>("TH2TTCluster_Position");
  HistoName = "Cluster_Endcap_Bw_XY";
  Cluster_Endcap_Bw_XY = iBooker.book2D(HistoName,
                                        HistoName,
                                        psTTCluster_Endcap_Bw_XY.getParameter<int32_t>("Nbinsx"),
                                        psTTCluster_Endcap_Bw_XY.getParameter<double>("xmin"),
                                        psTTCluster_Endcap_Bw_XY.getParameter<double>("xmax"),
                                        psTTCluster_Endcap_Bw_XY.getParameter<int32_t>("Nbinsy"),
                                        psTTCluster_Endcap_Bw_XY.getParameter<double>("ymin"),
                                        psTTCluster_Endcap_Bw_XY.getParameter<double>("ymax"));
  Cluster_Endcap_Bw_XY->setAxisTitle("L1 Cluster Backward Endcap position x [cm]", 1);
  Cluster_Endcap_Bw_XY->setAxisTitle("L1 Cluster Backward Endcap position y [cm]", 2);

  // TTCluster #rho vs. z
  edm::ParameterSet psTTCluster_RZ = conf_.getParameter<edm::ParameterSet>("TH2TTCluster_RZ");
  HistoName = "Cluster_RZ";
  Cluster_RZ = iBooker.book2D(HistoName,
                              HistoName,
                              psTTCluster_RZ.getParameter<int32_t>("Nbinsx"),
                              psTTCluster_RZ.getParameter<double>("xmin"),
                              psTTCluster_RZ.getParameter<double>("xmax"),
                              psTTCluster_RZ.getParameter<int32_t>("Nbinsy"),
                              psTTCluster_RZ.getParameter<double>("ymin"),
                              psTTCluster_RZ.getParameter<double>("ymax"));
  Cluster_RZ->setAxisTitle("L1 Cluster position z [cm]", 1);
  Cluster_RZ->setAxisTitle("L1 Cluster position #rho [cm]", 2);

}  // end of method

DEFINE_FWK_MODULE(OuterTrackerMonitorTTCluster);
