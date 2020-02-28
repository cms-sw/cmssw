// -*- C++ -*-
//
// Package:    SiOuterTracker
// Class:      SiOuterTracker
//
/**\class SiOuterTracker OuterTrackerMonitorTTStub.cc
 DQM/SiOuterTracker/plugins/OuterTrackerMonitorTTStub.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Isis Marina Van Parijs
//         Created:  Fri, 24 Oct 2014 12:31:31 GMT
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "DQM/SiOuterTracker/interface/OuterTrackerMonitorTTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// constructors and destructor
OuterTrackerMonitorTTStub::OuterTrackerMonitorTTStub(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  // now do what ever initialization is needed
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTStubsToken_ =
      consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTStubs"));
}

OuterTrackerMonitorTTStub::~OuterTrackerMonitorTTStub() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// member functions

// ------------ method called for each event  ------------
void OuterTrackerMonitorTTStub::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  /// Track Trigger Stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(tagTTStubsToken_, Phase2TrackerDigiTTStubHandle);

  /// Geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  const TrackerTopology *tTopo;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tTopo = tTopoHandle.product();

  edm::ESHandle<TrackerGeometry> tGeometryHandle;
  const TrackerGeometry *theTrackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeometryHandle);
  theTrackerGeometry = tGeometryHandle.product();

  /// Loop over input Stubs
  typename edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator inputIter;
  typename edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_>>::const_iterator contentIter;
  // Adding protection
  if (!Phase2TrackerDigiTTStubHandle.isValid())
    return;

  for (inputIter = Phase2TrackerDigiTTStubHandle->begin(); inputIter != Phase2TrackerDigiTTStubHandle->end();
       ++inputIter) {
    for (contentIter = inputIter->begin(); contentIter != inputIter->end(); ++contentIter) {
      /// Make reference stub
      edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>> tempStubRef =
          edmNew::makeRefTo(Phase2TrackerDigiTTStubHandle, contentIter);

      /// Get det ID (place of the stub)
      //  tempStubRef->getDetId() gives the stackDetId, not rawId
      DetId detIdStub = theTrackerGeometry->idToDet((tempStubRef->clusterRef(0))->getDetId())->geographicalId();

      /// Get trigger displacement/offset
      double displStub = tempStubRef->rawBend();
      double offsetStub = tempStubRef->bendOffset();

      /// Define position stub by position inner cluster
      MeasurementPoint mp = (tempStubRef->clusterRef(0))->findAverageLocalCoordinates();
      const GeomDet *theGeomDet = theTrackerGeometry->idToDet(detIdStub);
      Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(mp));

      Stub_Eta->Fill(posStub.eta());
      Stub_Phi->Fill(posStub.phi());
      Stub_R->Fill(posStub.perp());
      Stub_RZ->Fill(posStub.z(), posStub.perp());

      if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TOB)) {  // Phase 2 Outer Tracker Barrel
        Stub_Barrel->Fill(tTopo->layer(detIdStub));
        Stub_Barrel_XY->Fill(posStub.x(), posStub.y());
        Stub_Barrel_W->Fill(tTopo->layer(detIdStub), displStub - offsetStub);
        Stub_Barrel_O->Fill(tTopo->layer(detIdStub), offsetStub);
      } else if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TID)) {  // Phase 2 Outer Tracker Endcap
        int disc = tTopo->layer(detIdStub);                                          // returns wheel
        int ring = tTopo->tidRing(detIdStub);
        Stub_Endcap_Disc->Fill(disc);
        Stub_Endcap_Ring->Fill(ring);
        Stub_Endcap_Disc_W->Fill(disc, displStub - offsetStub);
        Stub_Endcap_Ring_W->Fill(ring, displStub - offsetStub);
        Stub_Endcap_Disc_O->Fill(disc, offsetStub);
        Stub_Endcap_Ring_O->Fill(ring, offsetStub);

        if (posStub.z() > 0) {
          Stub_Endcap_Fw_XY->Fill(posStub.x(), posStub.y());
          Stub_Endcap_Disc_Fw->Fill(disc);
          Stub_Endcap_Ring_Fw[disc - 1]->Fill(ring);
          Stub_Endcap_Ring_W_Fw[disc - 1]->Fill(ring, displStub - offsetStub);
          Stub_Endcap_Ring_O_Fw[disc - 1]->Fill(ring, offsetStub);
        } else {
          Stub_Endcap_Bw_XY->Fill(posStub.x(), posStub.y());
          Stub_Endcap_Disc_Bw->Fill(disc);
          Stub_Endcap_Ring_Bw[disc - 1]->Fill(ring);
          Stub_Endcap_Ring_W_Bw[disc - 1]->Fill(ring, displStub - offsetStub);
          Stub_Endcap_Ring_O_Bw[disc - 1]->Fill(ring, offsetStub);
        }
      }
    }
  }
}  // end of method

// ------------ method called when starting to processes a run  ------------
void OuterTrackerMonitorTTStub::bookHistograms(DQMStore::IBooker &iBooker,
                                               edm::Run const &run,
                                               edm::EventSetup const &es) {
  std::string HistoName;
  const int numDiscs = 5;
  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/Position");

  edm::ParameterSet psTTStub_Barrel_XY = conf_.getParameter<edm::ParameterSet>("TH2TTStub_Position");
  HistoName = "Stub_Barrel_XY";
  Stub_Barrel_XY = iBooker.book2D(HistoName,
                                  HistoName,
                                  psTTStub_Barrel_XY.getParameter<int32_t>("Nbinsx"),
                                  psTTStub_Barrel_XY.getParameter<double>("xmin"),
                                  psTTStub_Barrel_XY.getParameter<double>("xmax"),
                                  psTTStub_Barrel_XY.getParameter<int32_t>("Nbinsy"),
                                  psTTStub_Barrel_XY.getParameter<double>("ymin"),
                                  psTTStub_Barrel_XY.getParameter<double>("ymax"));
  Stub_Barrel_XY->setAxisTitle("L1 Stub Barrel position x [cm]", 1);
  Stub_Barrel_XY->setAxisTitle("L1 Stub Barrel position y [cm]", 2);

  edm::ParameterSet psTTStub_Endcap_Fw_XY = conf_.getParameter<edm::ParameterSet>("TH2TTStub_Position");
  HistoName = "Stub_Endcap_Fw_XY";
  Stub_Endcap_Fw_XY = iBooker.book2D(HistoName,
                                     HistoName,
                                     psTTStub_Endcap_Fw_XY.getParameter<int32_t>("Nbinsx"),
                                     psTTStub_Endcap_Fw_XY.getParameter<double>("xmin"),
                                     psTTStub_Endcap_Fw_XY.getParameter<double>("xmax"),
                                     psTTStub_Endcap_Fw_XY.getParameter<int32_t>("Nbinsy"),
                                     psTTStub_Endcap_Fw_XY.getParameter<double>("ymin"),
                                     psTTStub_Endcap_Fw_XY.getParameter<double>("ymax"));
  Stub_Endcap_Fw_XY->setAxisTitle("L1 Stub Endcap position x [cm]", 1);
  Stub_Endcap_Fw_XY->setAxisTitle("L1 Stub Endcap position y [cm]", 2);

  edm::ParameterSet psTTStub_Endcap_Bw_XY = conf_.getParameter<edm::ParameterSet>("TH2TTStub_Position");
  HistoName = "Stub_Endcap_Bw_XY";
  Stub_Endcap_Bw_XY = iBooker.book2D(HistoName,
                                     HistoName,
                                     psTTStub_Endcap_Bw_XY.getParameter<int32_t>("Nbinsx"),
                                     psTTStub_Endcap_Bw_XY.getParameter<double>("xmin"),
                                     psTTStub_Endcap_Bw_XY.getParameter<double>("xmax"),
                                     psTTStub_Endcap_Bw_XY.getParameter<int32_t>("Nbinsy"),
                                     psTTStub_Endcap_Bw_XY.getParameter<double>("ymin"),
                                     psTTStub_Endcap_Bw_XY.getParameter<double>("ymax"));
  Stub_Endcap_Bw_XY->setAxisTitle("L1 Stub Endcap position x [cm]", 1);
  Stub_Endcap_Bw_XY->setAxisTitle("L1 Stub Endcap position y [cm]", 2);

  // TTStub #rho vs. z
  edm::ParameterSet psTTStub_RZ = conf_.getParameter<edm::ParameterSet>("TH2TTStub_RZ");
  HistoName = "Stub_RZ";
  Stub_RZ = iBooker.book2D(HistoName,
                           HistoName,
                           psTTStub_RZ.getParameter<int32_t>("Nbinsx"),
                           psTTStub_RZ.getParameter<double>("xmin"),
                           psTTStub_RZ.getParameter<double>("xmax"),
                           psTTStub_RZ.getParameter<int32_t>("Nbinsy"),
                           psTTStub_RZ.getParameter<double>("ymin"),
                           psTTStub_RZ.getParameter<double>("ymax"));
  Stub_RZ->setAxisTitle("L1 Stub position z [cm]", 1);
  Stub_RZ->setAxisTitle("L1 Stub position #rho [cm]", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs");
  // TTStub eta
  edm::ParameterSet psTTStub_Eta = conf_.getParameter<edm::ParameterSet>("TH1TTStub_Eta");
  HistoName = "Stub_Eta";
  Stub_Eta = iBooker.book1D(HistoName,
                            HistoName,
                            psTTStub_Eta.getParameter<int32_t>("Nbinsx"),
                            psTTStub_Eta.getParameter<double>("xmin"),
                            psTTStub_Eta.getParameter<double>("xmax"));
  Stub_Eta->setAxisTitle("#eta", 1);
  Stub_Eta->setAxisTitle("# L1 Stubs ", 2);

  // TTStub phi
  edm::ParameterSet psTTStub_Phi = conf_.getParameter<edm::ParameterSet>("TH1TTStub_Phi");
  HistoName = "Stub_Phi";
  Stub_Phi = iBooker.book1D(HistoName,
                            HistoName,
                            psTTStub_Phi.getParameter<int32_t>("Nbinsx"),
                            psTTStub_Phi.getParameter<double>("xmin"),
                            psTTStub_Phi.getParameter<double>("xmax"));
  Stub_Phi->setAxisTitle("#phi", 1);
  Stub_Phi->setAxisTitle("# L1 Stubs ", 2);

  // TTStub R
  edm::ParameterSet psTTStub_R = conf_.getParameter<edm::ParameterSet>("TH1TTStub_R");
  HistoName = "Stub_R";
  Stub_R = iBooker.book1D(HistoName,
                          HistoName,
                          psTTStub_R.getParameter<int32_t>("Nbinsx"),
                          psTTStub_R.getParameter<double>("xmin"),
                          psTTStub_R.getParameter<double>("xmax"));
  Stub_R->setAxisTitle("R", 1);
  Stub_R->setAxisTitle("# L1 Stubs ", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/NStubs");
  // TTStub barrel stack
  edm::ParameterSet psTTStub_Barrel = conf_.getParameter<edm::ParameterSet>("TH1TTStub_Layers");
  HistoName = "NStubs_Barrel";
  Stub_Barrel = iBooker.book1D(HistoName,
                               HistoName,
                               psTTStub_Barrel.getParameter<int32_t>("Nbinsx"),
                               psTTStub_Barrel.getParameter<double>("xmin"),
                               psTTStub_Barrel.getParameter<double>("xmax"));
  Stub_Barrel->setAxisTitle("Barrel Layer", 1);
  Stub_Barrel->setAxisTitle("# L1 Stubs ", 2);

  // TTStub Endcap stack
  edm::ParameterSet psTTStub_ECDisc = conf_.getParameter<edm::ParameterSet>("TH1TTStub_Discs");
  HistoName = "NStubs_Endcap_Disc";
  Stub_Endcap_Disc = iBooker.book1D(HistoName,
                                    HistoName,
                                    psTTStub_ECDisc.getParameter<int32_t>("Nbinsx"),
                                    psTTStub_ECDisc.getParameter<double>("xmin"),
                                    psTTStub_ECDisc.getParameter<double>("xmax"));
  Stub_Endcap_Disc->setAxisTitle("Endcap Disc", 1);
  Stub_Endcap_Disc->setAxisTitle("# L1 Stubs ", 2);

  // TTStub Endcap stack
  HistoName = "NStubs_Endcap_Disc_Fw";
  Stub_Endcap_Disc_Fw = iBooker.book1D(HistoName,
                                       HistoName,
                                       psTTStub_ECDisc.getParameter<int32_t>("Nbinsx"),
                                       psTTStub_ECDisc.getParameter<double>("xmin"),
                                       psTTStub_ECDisc.getParameter<double>("xmax"));
  Stub_Endcap_Disc_Fw->setAxisTitle("Forward Endcap Disc", 1);
  Stub_Endcap_Disc_Fw->setAxisTitle("# L1 Stubs ", 2);

  // TTStub Endcap stack
  HistoName = "NStubs_Endcap_Disc_Bw";
  Stub_Endcap_Disc_Bw = iBooker.book1D(HistoName,
                                       HistoName,
                                       psTTStub_ECDisc.getParameter<int32_t>("Nbinsx"),
                                       psTTStub_ECDisc.getParameter<double>("xmin"),
                                       psTTStub_ECDisc.getParameter<double>("xmax"));
  Stub_Endcap_Disc_Bw->setAxisTitle("Backward Endcap Disc", 1);
  Stub_Endcap_Disc_Bw->setAxisTitle("# L1 Stubs ", 2);

  edm::ParameterSet psTTStub_ECRing = conf_.getParameter<edm::ParameterSet>("TH1TTStub_Rings");
  HistoName = "NStubs_Endcap_Ring";
  Stub_Endcap_Ring = iBooker.book1D(HistoName,
                                    HistoName,
                                    psTTStub_ECRing.getParameter<int32_t>("Nbinsx"),
                                    psTTStub_ECRing.getParameter<double>("xmin"),
                                    psTTStub_ECRing.getParameter<double>("xmax"));
  Stub_Endcap_Ring->setAxisTitle("Endcap Ring", 1);
  Stub_Endcap_Ring->setAxisTitle("# L1 Stubs ", 2);

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "NStubs_Disc+" + std::to_string(i + 1);
    // TTStub Endcap stack
    Stub_Endcap_Ring_Fw[i] = iBooker.book1D(HistoName,
                                            HistoName,
                                            psTTStub_ECRing.getParameter<int32_t>("Nbinsx"),
                                            psTTStub_ECRing.getParameter<double>("xmin"),
                                            psTTStub_ECRing.getParameter<double>("xmax"));
    Stub_Endcap_Ring_Fw[i]->setAxisTitle("Endcap Ring", 1);
    Stub_Endcap_Ring_Fw[i]->setAxisTitle("# L1 Stubs ", 2);
  }

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "NStubs_Disc-" + std::to_string(i + 1);
    // TTStub Endcap stack
    Stub_Endcap_Ring_Bw[i] = iBooker.book1D(HistoName,
                                            HistoName,
                                            psTTStub_ECRing.getParameter<int32_t>("Nbinsx"),
                                            psTTStub_ECRing.getParameter<double>("xmin"),
                                            psTTStub_ECRing.getParameter<double>("xmax"));
    Stub_Endcap_Ring_Bw[i]->setAxisTitle("Endcap Ring", 1);
    Stub_Endcap_Ring_Bw[i]->setAxisTitle("# L1 Stubs ", 2);
  }

  // TTStub displ/offset
  edm::ParameterSet psTTStub_Barrel_2D = conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf_Layer");
  edm::ParameterSet psTTStub_ECDisc_2D = conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf_Disc");
  edm::ParameterSet psTTStub_ECRing_2D = conf_.getParameter<edm::ParameterSet>("TH2TTStub_DisOf_Ring");

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/Width");
  HistoName = "Stub_Width_Barrel";
  Stub_Barrel_W = iBooker.book2D(HistoName,
                                 HistoName,
                                 psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsx"),
                                 psTTStub_Barrel_2D.getParameter<double>("xmin"),
                                 psTTStub_Barrel_2D.getParameter<double>("xmax"),
                                 psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsy"),
                                 psTTStub_Barrel_2D.getParameter<double>("ymin"),
                                 psTTStub_Barrel_2D.getParameter<double>("ymax"));
  Stub_Barrel_W->setAxisTitle("Barrel Layer", 1);
  Stub_Barrel_W->setAxisTitle("Displacement - Offset", 2);

  HistoName = "Stub_Width_Endcap_Disc";
  Stub_Endcap_Disc_W = iBooker.book2D(HistoName,
                                      HistoName,
                                      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsx"),
                                      psTTStub_ECDisc_2D.getParameter<double>("xmin"),
                                      psTTStub_ECDisc_2D.getParameter<double>("xmax"),
                                      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsy"),
                                      psTTStub_ECDisc_2D.getParameter<double>("ymin"),
                                      psTTStub_ECDisc_2D.getParameter<double>("ymax"));
  Stub_Endcap_Disc_W->setAxisTitle("Endcap Disc", 1);
  Stub_Endcap_Disc_W->setAxisTitle("Displacement - Offset", 2);

  HistoName = "Stub_Width_Endcap_Ring";
  Stub_Endcap_Ring_W = iBooker.book2D(HistoName,
                                      HistoName,
                                      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
                                      psTTStub_ECRing_2D.getParameter<double>("xmin"),
                                      psTTStub_ECRing_2D.getParameter<double>("xmax"),
                                      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
                                      psTTStub_ECRing_2D.getParameter<double>("ymin"),
                                      psTTStub_ECRing_2D.getParameter<double>("ymax"));
  Stub_Endcap_Ring_W->setAxisTitle("Endcap Ring", 1);
  Stub_Endcap_Ring_W->setAxisTitle("Trigger Offset", 2);

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "Stub_Width_Disc+" + std::to_string(i + 1);
    Stub_Endcap_Ring_W_Fw[i] = iBooker.book2D(HistoName,
                                              HistoName,
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmin"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmax"),
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymin"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymax"));
    Stub_Endcap_Ring_W_Fw[i]->setAxisTitle("Endcap Ring", 1);
    Stub_Endcap_Ring_W_Fw[i]->setAxisTitle("Displacement - Offset", 2);
  }

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "Stub_Width_Disc-" + std::to_string(i + 1);
    Stub_Endcap_Ring_W_Bw[i] = iBooker.book2D(HistoName,
                                              HistoName,
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmin"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmax"),
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymin"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymax"));
    Stub_Endcap_Ring_W_Bw[i]->setAxisTitle("Endcap Ring", 1);
    Stub_Endcap_Ring_W_Bw[i]->setAxisTitle("Displacement - Offset", 2);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/Offset");
  HistoName = "Stub_Offset_Barrel";
  Stub_Barrel_O = iBooker.book2D(HistoName,
                                 HistoName,
                                 psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsx"),
                                 psTTStub_Barrel_2D.getParameter<double>("xmin"),
                                 psTTStub_Barrel_2D.getParameter<double>("xmax"),
                                 psTTStub_Barrel_2D.getParameter<int32_t>("Nbinsy"),
                                 psTTStub_Barrel_2D.getParameter<double>("ymin"),
                                 psTTStub_Barrel_2D.getParameter<double>("ymax"));
  Stub_Barrel_O->setAxisTitle("Barrel Layer", 1);
  Stub_Barrel_O->setAxisTitle("Trigger Offset", 2);

  HistoName = "Stub_Offset_Endcap_Disc";
  Stub_Endcap_Disc_O = iBooker.book2D(HistoName,
                                      HistoName,
                                      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsx"),
                                      psTTStub_ECDisc_2D.getParameter<double>("xmin"),
                                      psTTStub_ECDisc_2D.getParameter<double>("xmax"),
                                      psTTStub_ECDisc_2D.getParameter<int32_t>("Nbinsy"),
                                      psTTStub_ECDisc_2D.getParameter<double>("ymin"),
                                      psTTStub_ECDisc_2D.getParameter<double>("ymax"));
  Stub_Endcap_Disc_O->setAxisTitle("Endcap Disc", 1);
  Stub_Endcap_Disc_O->setAxisTitle("Trigger Offset", 2);

  HistoName = "Stub_Offset_Endcap_Ring";
  Stub_Endcap_Ring_O = iBooker.book2D(HistoName,
                                      HistoName,
                                      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
                                      psTTStub_ECRing_2D.getParameter<double>("xmin"),
                                      psTTStub_ECRing_2D.getParameter<double>("xmax"),
                                      psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
                                      psTTStub_ECRing_2D.getParameter<double>("ymin"),
                                      psTTStub_ECRing_2D.getParameter<double>("ymax"));
  Stub_Endcap_Ring_O->setAxisTitle("Endcap Ring", 1);
  Stub_Endcap_Ring_O->setAxisTitle("Trigger Offset", 2);

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "Stub_Offset_Disc+" + std::to_string(i + 1);
    Stub_Endcap_Ring_O_Fw[i] = iBooker.book2D(HistoName,
                                              HistoName,
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmin"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmax"),
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymin"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymax"));
    Stub_Endcap_Ring_O_Fw[i]->setAxisTitle("Endcap Ring", 1);
    Stub_Endcap_Ring_O_Fw[i]->setAxisTitle("Trigger Offset", 2);
  }

  for (int i = 0; i < numDiscs; i++) {
    HistoName = "Stub_Offset_Disc-" + std::to_string(i + 1);
    Stub_Endcap_Ring_O_Bw[i] = iBooker.book2D(HistoName,
                                              HistoName,
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsx"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmin"),
                                              psTTStub_ECRing_2D.getParameter<double>("xmax"),
                                              psTTStub_ECRing_2D.getParameter<int32_t>("Nbinsy"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymin"),
                                              psTTStub_ECRing_2D.getParameter<double>("ymax"));
    Stub_Endcap_Ring_O_Bw[i]->setAxisTitle("Endcap Ring", 1);
    Stub_Endcap_Ring_O_Bw[i]->setAxisTitle("Trigger Offset", 2);
  }
}

DEFINE_FWK_MODULE(OuterTrackerMonitorTTStub);
