// -*- C++ -*-
//
// Package:    SiOuterTracker
// Class:      SiOuterTracker
//
/**\class SiOuterTracker Phase2OTMonitorTTStub.cc
 DQM/SiOuterTracker/plugins/Phase2OTMonitorTTStub.cc

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
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

class Phase2OTMonitorTTStub : public DQMEDAnalyzer {
public:
  explicit Phase2OTMonitorTTStub(const edm::ParameterSet &);
  ~Phase2OTMonitorTTStub() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  // TTStub stacks
  // Global position of the stubs
  MonitorElement *Stub_Barrel_XY = nullptr;     // TTStub barrel y vs x
  MonitorElement *Stub_Endcap_Fw_XY = nullptr;  // TTStub Forward Endcap y vs. x
  MonitorElement *Stub_Endcap_Bw_XY = nullptr;  // TTStub Backward Endcap y vs. x
  MonitorElement *Stub_RZ = nullptr;            // TTStub #rho vs. z
  MonitorElement *CrackOverview = nullptr;      // Cosmic rack: TTStub layer vs module

  // Number of stubs
  MonitorElement *Stub_Barrel = nullptr;                                                   // TTStub per layer
  MonitorElement *Stub_Endcap_Disc = nullptr;                                              // TTStubs per disc
  MonitorElement *Stub_Endcap_Disc_Fw = nullptr;                                           // TTStub per disc
  MonitorElement *Stub_Endcap_Disc_Bw = nullptr;                                           // TTStub per disc
  MonitorElement *Stub_Endcap_Ring = nullptr;                                              // TTStubs per ring
  MonitorElement *Stub_Endcap_Ring_Fw[trklet::N_DISK] = {};  // TTStubs per EC ring
  MonitorElement *Stub_Endcap_Ring_Bw[trklet::N_DISK] = {};  // TTStub per EC ring

  // Stub distribution
  MonitorElement *Stub_Eta = nullptr;     // TTstub eta distribution
  MonitorElement *Stub_Phi = nullptr;     // TTstub phi distribution
  MonitorElement *Stub_R = nullptr;       // TTstub r distribution
  MonitorElement *Stub_bendFE = nullptr;  // TTstub trigger bend
  MonitorElement *Stub_bendBE = nullptr;  // TTstub hardware bend
  MonitorElement *Stub_isPS = nullptr;    // is this stub a PS module?

  // STUB Displacement - offset
  MonitorElement *Stub_Barrel_W = nullptr;       // TTstub Pos-Corr Displacement (layer)
  MonitorElement *Stub_Barrel_O = nullptr;       // TTStub Offset (layer)
  MonitorElement *Stub_Endcap_Disc_W = nullptr;  // TTstub Pos-Corr Displacement (disc)
  MonitorElement *Stub_Endcap_Disc_O = nullptr;  // TTStub Offset (disc)
  MonitorElement *Stub_Endcap_Ring_W = nullptr;  // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement *Stub_Endcap_Ring_O = nullptr;  // TTStub Offset (EC ring)
  MonitorElement *Stub_Endcap_Ring_W_Fw[trklet::N_DISK] = {};  // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement *Stub_Endcap_Ring_O_Fw[trklet::N_DISK] = {};  // TTStub Offset (EC ring)
  MonitorElement *Stub_Endcap_Ring_W_Bw[trklet::N_DISK] = {};  // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement *Stub_Endcap_Ring_O_Bw[trklet::N_DISK] = {};  // TTStub Offset (EC ring)

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> tagTTStubsToken_;
  std::string topFolderName_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry *tkGeom_ = nullptr;
  const TrackerTopology *tTopo_ = nullptr;
};

// constructors and destructor
Phase2OTMonitorTTStub::Phase2OTMonitorTTStub(const edm::ParameterSet &iConfig)
    : conf_(iConfig),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  // now do what ever initialization is needed
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTStubsToken_ =
      consumes<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTStubs"));
}

Phase2OTMonitorTTStub::~Phase2OTMonitorTTStub() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void Phase2OTMonitorTTStub::dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {
  tkGeom_ = &(iSetup.getData(geomToken_));
  tTopo_ = &(iSetup.getData(topoToken_));
}
// member functions

// ------------ method called for each event  ------------
void Phase2OTMonitorTTStub::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  /// Track Trigger Stubs
  edm::Handle<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>> Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken(tagTTStubsToken_, Phase2TrackerDigiTTStubHandle);

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
      DetId detIdStub = tkGeom_->idToDet((tempStubRef->clusterRef(0))->getDetId())->geographicalId();

      /// Get trigger displacement/offset
      double rawBend = tempStubRef->rawBend();
      double bendOffset = tempStubRef->bendOffset();

      // Get module
      unsigned int module = tTopo_->module(detIdStub);
      // CRACK is viewed from behind, so to align plots with what is seen in real life, modules are flipped
      if (CrackOverview)
        module = std::abs(int(module - 13));
      /// Define position stub by position inner cluster
      MeasurementPoint mp = (tempStubRef->clusterRef(0))->findAverageLocalCoordinates();
      const GeomDet *theGeomDet = tkGeom_->idToDet(detIdStub);
      Global3DPoint posStub = theGeomDet->surface().toGlobal(theGeomDet->topology().localPosition(mp));

      Stub_Eta->Fill(posStub.eta());
      Stub_Phi->Fill(posStub.phi());
      Stub_R->Fill(posStub.perp());
      Stub_RZ->Fill(posStub.z(), posStub.perp());
      Stub_bendFE->Fill(tempStubRef->bendFE());
      Stub_bendBE->Fill(tempStubRef->bendBE());
      Stub_isPS->Fill(tempStubRef->moduleTypePS());
      if (CrackOverview)
        CrackOverview->Fill(module, tTopo_->getOTLayerNumber(detIdStub) + 0.05 - (module % 2 * 0.1));

      if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TOB)) {  // Phase 2 Outer Tracker Barrel
        Stub_Barrel->Fill(tTopo_->layer(detIdStub));
        Stub_Barrel_XY->Fill(posStub.x(), posStub.y());
        Stub_Barrel_W->Fill(tTopo_->layer(detIdStub), rawBend - bendOffset);
        Stub_Barrel_O->Fill(tTopo_->layer(detIdStub), bendOffset);
      } else if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TID)) {  // Phase 2 Outer Tracker Endcap
        int disc = tTopo_->layer(detIdStub);                                         // returns wheel
        int ring = tTopo_->tidRing(detIdStub);
        Stub_Endcap_Disc->Fill(disc);
        Stub_Endcap_Ring->Fill(ring);
        Stub_Endcap_Disc_W->Fill(disc, rawBend - bendOffset);
        Stub_Endcap_Ring_W->Fill(ring, rawBend - bendOffset);
        Stub_Endcap_Disc_O->Fill(disc, bendOffset);
        Stub_Endcap_Ring_O->Fill(ring, bendOffset);

        if (posStub.z() > 0) {
          Stub_Endcap_Fw_XY->Fill(posStub.x(), posStub.y());
          Stub_Endcap_Disc_Fw->Fill(disc);
          Stub_Endcap_Ring_Fw[disc - 1]->Fill(ring);
          Stub_Endcap_Ring_W_Fw[disc - 1]->Fill(ring, rawBend - bendOffset);
          Stub_Endcap_Ring_O_Fw[disc - 1]->Fill(ring, bendOffset);
        } else {
          Stub_Endcap_Bw_XY->Fill(posStub.x(), posStub.y());
          Stub_Endcap_Disc_Bw->Fill(disc);
          Stub_Endcap_Ring_Bw[disc - 1]->Fill(ring);
          Stub_Endcap_Ring_W_Bw[disc - 1]->Fill(ring, rawBend - bendOffset);
          Stub_Endcap_Ring_O_Bw[disc - 1]->Fill(ring, bendOffset);
        }
      }
    }
  }
}  // end of method

void Phase2OTMonitorTTStub::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &run, edm::EventSetup const &es) {
  using namespace phase2tkutil;

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/Position");
  Stub_Barrel_XY    = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Barrel_XY"),    iBooker);
  Stub_Endcap_Fw_XY = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Endcap_Fw_XY"), iBooker);
  Stub_Endcap_Bw_XY = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Endcap_Bw_XY"), iBooker);
  Stub_RZ           = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_RZ"),            iBooker);

  // CRACK ONLY: module vs layer
  edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("CrackOverview");
  if (Parameters.getParameter<bool>("switch")) {
    CrackOverview = iBooker.book2DPoly(Parameters.getParameter<std::string>("name"),
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

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs");
  Stub_Eta    = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Eta"),    iBooker);
  Stub_Phi    = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Phi"),    iBooker);
  Stub_R      = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_R"),      iBooker);
  Stub_bendFE = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_bendFE"), iBooker);
  Stub_bendBE = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_bendBE"), iBooker);
  Stub_isPS   = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_isPS"),   iBooker);

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/NStubs");
  Stub_Barrel         = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Barrel"),         iBooker);
  Stub_Endcap_Disc    = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Endcap_Disc"),    iBooker);
  Stub_Endcap_Disc_Fw = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Endcap_Disc_Fw"), iBooker);
  Stub_Endcap_Disc_Bw = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Endcap_Disc_Bw"), iBooker);
  Stub_Endcap_Ring    = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Endcap_Ring"),    iBooker);

  for (int i = 0; i < static_cast<int>(trklet::N_DISK); i++) {
    Stub_Endcap_Ring_Fw[i] = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Disc_Fw_" + std::to_string(i + 1)), iBooker);
    Stub_Endcap_Ring_Bw[i] = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("NStubs_Disc_Bw_" + std::to_string(i + 1)), iBooker);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/Width");
  Stub_Barrel_W      = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Width_Barrel"),      iBooker);
  Stub_Endcap_Disc_W = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Width_Endcap_Disc"), iBooker);
  Stub_Endcap_Ring_W = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Width_Endcap_Ring"), iBooker);
  for (int i = 0; i < static_cast<int>(trklet::N_DISK); i++) {
    Stub_Endcap_Ring_W_Fw[i] = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Width_Disc_Fw_"  + std::to_string(i + 1)), iBooker);
    Stub_Endcap_Ring_W_Bw[i] = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Width_Disc_Bw_"  + std::to_string(i + 1)), iBooker);
  }

  iBooker.setCurrentFolder(topFolderName_ + "/Stubs/Offset");
  Stub_Barrel_O      = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Offset_Barrel"),      iBooker);
  Stub_Endcap_Disc_O = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Offset_Endcap_Disc"), iBooker);
  Stub_Endcap_Ring_O = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Offset_Endcap_Ring"), iBooker);
  for (int i = 0; i < static_cast<int>(trklet::N_DISK); i++) {
    Stub_Endcap_Ring_O_Fw[i] = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Offset_Disc_Fw_" + std::to_string(i + 1)), iBooker);
    Stub_Endcap_Ring_O_Bw[i] = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("Stub_Offset_Disc_Bw_" + std::to_string(i + 1)), iBooker);
  }
}

void Phase2OTMonitorTTStub::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // Position
  phase2tkutil::add2DDesc(desc, "Stub_Barrel_XY",    "Stub_Barrel_XY",    "L1 Stub Barrel position x [cm]", "L1 Stub Barrel position y [cm]", 960, -120, 120, 960, -120, 120);
  phase2tkutil::add2DDesc(desc, "Stub_Endcap_Fw_XY", "Stub_Endcap_Fw_XY", "L1 Stub Endcap position x [cm]", "L1 Stub Endcap position y [cm]", 960, -120, 120, 960, -120, 120);
  phase2tkutil::add2DDesc(desc, "Stub_Endcap_Bw_XY", "Stub_Endcap_Bw_XY", "L1 Stub Endcap position x [cm]", "L1 Stub Endcap position y [cm]", 960, -120, 120, 960, -120, 120);
  phase2tkutil::add2DDesc(desc, "Stub_RZ",            "Stub_RZ",            "L1 Stub position z [cm]",        "L1 Stub position #rho [cm]",    900, -300, 300, 900, 0, 120);

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Crack_Overview_Stubs");
    psd0.add<std::string>("title", "Crack_Overview_stubs;Module;Layer");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", false);
    psd0.add<double>("xmax", 13.0);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 7.5);
    desc.add<edm::ParameterSetDescription>("CrackOverview", psd0);
  }

  // Stub distributions
  phase2tkutil::add1DDesc(desc, "Stub_Eta",    "Stub_Eta",    "#eta",          "# L1 Stubs", 45, -5,     5);
  phase2tkutil::add1DDesc(desc, "Stub_Phi",    "Stub_Phi",    "#phi",          "# L1 Stubs", 60, -3.5,   3.5);
  phase2tkutil::add1DDesc(desc, "Stub_R",      "Stub_R",      "R",             "# L1 Stubs", 45,  0,     120);
  phase2tkutil::add1DDesc(desc, "Stub_bendFE", "Stub_bendFE", "Trigger bend",  "# L1 Stubs", 69, -8.625, 8.625);
  phase2tkutil::add1DDesc(desc, "Stub_bendBE", "Stub_bendBE", "Hardware bend", "# L1 Stubs", 69, -8.625, 8.625);
  phase2tkutil::add1DDesc(desc, "Stub_isPS",   "Stub_isPS",   "Is PS?",        "# L1 Stubs",  2,  0,     2);

  // NStubs
  phase2tkutil::add1DDesc(desc, "NStubs_Barrel",         "NStubs_Barrel",         "Barrel Layer",         "# L1 Stubs",  7, 0.5, 7.5);
  phase2tkutil::add1DDesc(desc, "NStubs_Endcap_Disc",    "NStubs_Endcap_Disc",    "Endcap Disc",          "# L1 Stubs",  6, 0.5, 6.5);
  phase2tkutil::add1DDesc(desc, "NStubs_Endcap_Disc_Fw", "NStubs_Endcap_Disc_Fw", "Forward Endcap Disc",  "# L1 Stubs",  6, 0.5, 6.5);
  phase2tkutil::add1DDesc(desc, "NStubs_Endcap_Disc_Bw", "NStubs_Endcap_Disc_Bw", "Backward Endcap Disc", "# L1 Stubs",  6, 0.5, 6.5);
  phase2tkutil::add1DDesc(desc, "NStubs_Endcap_Ring",    "NStubs_Endcap_Ring",    "Endcap Ring",          "# L1 Stubs", 16, 0.5, 16.5);

  // Width
  phase2tkutil::add2DDesc(desc, "Stub_Width_Barrel",      "Stub_Width_Barrel",      "Barrel Layer", "Displacement - Offset",  6, 0.5,  6.5, 43, -10.75, 10.75);
  phase2tkutil::add2DDesc(desc, "Stub_Width_Endcap_Disc", "Stub_Width_Endcap_Disc", "Endcap Disc",  "Displacement - Offset",  5, 0.5,  5.5, 43, -10.75, 10.75);
  phase2tkutil::add2DDesc(desc, "Stub_Width_Endcap_Ring", "Stub_Width_Endcap_Ring", "Endcap Ring",  "Displacement - Offset", 16, 0.5, 16.5, 43, -10.75, 10.75);

  // Offset
  phase2tkutil::add2DDesc(desc, "Stub_Offset_Barrel",      "Stub_Offset_Barrel",      "Barrel Layer", "Trigger Offset",  6, 0.5,  6.5, 43, -10.75, 10.75);
  phase2tkutil::add2DDesc(desc, "Stub_Offset_Endcap_Disc", "Stub_Offset_Endcap_Disc", "Endcap Disc",  "Trigger Offset",  5, 0.5,  5.5, 43, -10.75, 10.75);
  phase2tkutil::add2DDesc(desc, "Stub_Offset_Endcap_Ring", "Stub_Offset_Endcap_Ring", "Endcap Ring",  "Trigger Offset", 16, 0.5, 16.5, 43, -10.75, 10.75);

  // Disc-specific ring histograms
  for (int i = 1; i <= static_cast<int>(trklet::N_DISK); i++) {
    const std::string si = std::to_string(i);
    phase2tkutil::add1DDesc(desc, "NStubs_Disc_Fw_" + si, "NStubs_Disc+" + si, "Endcap Ring", "# L1 Stubs", 16, 0.5, 16.5);
    phase2tkutil::add1DDesc(desc, "NStubs_Disc_Bw_" + si, "NStubs_Disc-" + si, "Endcap Ring", "# L1 Stubs", 16, 0.5, 16.5);
    phase2tkutil::add2DDesc(desc, "Stub_Width_Disc_Fw_"  + si, "Stub_Width_Disc+"  + si, "Endcap Ring", "Displacement - Offset", 16, 0.5, 16.5, 43, -10.75, 10.75);
    phase2tkutil::add2DDesc(desc, "Stub_Width_Disc_Bw_"  + si, "Stub_Width_Disc-"  + si, "Endcap Ring", "Displacement - Offset", 16, 0.5, 16.5, 43, -10.75, 10.75);
    phase2tkutil::add2DDesc(desc, "Stub_Offset_Disc_Fw_" + si, "Stub_Offset_Disc+" + si, "Endcap Ring", "Trigger Offset", 16, 0.5, 16.5, 43, -10.75, 10.75);
    phase2tkutil::add2DDesc(desc, "Stub_Offset_Disc_Bw_" + si, "Stub_Offset_Disc-" + si, "Endcap Ring", "Trigger Offset", 16, 0.5, 16.5, 43, -10.75, 10.75);
  }

  desc.add<std::string>("TopFolderName", "TrackerPhase2OTStub");
  desc.add<edm::InputTag>("TTStubs", edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  descriptions.add("Phase2OTMonitorTTStub", desc);
}

DEFINE_FWK_MODULE(Phase2OTMonitorTTStub);
