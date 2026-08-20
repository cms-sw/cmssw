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
// Edited: August 2026 by Lisa Juckett for dqm output restructure
//

// system include files
#include <memory>
#include <numeric>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
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

#include "Geometry/CommonTopologies/interface/GeomDet.h"
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
  MonitorElement *Stub_Barrel = nullptr;  // TTStub per layer

  // Stub distribution
  MonitorElement *Stub_Eta = nullptr;     // TTstub eta distribution
  MonitorElement *Stub_Phi = nullptr;     // TTstub phi distribution
  MonitorElement *Stub_R = nullptr;       // TTstub r distribution
  MonitorElement *Stub_bendFE = nullptr;  // TTstub trigger bend
  MonitorElement *Stub_bendBE = nullptr;  // TTstub hardware bend
  MonitorElement *Stub_isPS = nullptr;    // is this stub a PS module?

  // STUB Displacement - offset
  MonitorElement *Stub_Barrel_W = nullptr;  // TTstub Pos-Corr Displacement (layer)
  MonitorElement *Stub_Barrel_O = nullptr;  // TTStub Offset (layer)

private:
  struct TTStubMEs {
    MonitorElement *NStubs = nullptr;
    MonitorElement *NStubsByRing = nullptr;
    MonitorElement *NStubsByWheel = nullptr;
    MonitorElement *StubOffsetByRing = nullptr;
    MonitorElement *StubOffsetByWheel = nullptr;
    MonitorElement *StubWidthByRing = nullptr;
    MonitorElement *StubWidthByWheel = nullptr;
    unsigned int stubCounter = 0;
  };

  std::map<std::string, TTStubMEs> layerMEs_;
  enum Level { OT = 1, SUBSTRUCTURE, ENDCAP_SIDE, ENDCAP_RING, ENDCAP_WHEEL, LAYER };

  void bookLayerHistos(DQMStore::IBooker &ibooker, uint32_t det_id, std::string &subdir);
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
      int wheel = tTopo_->tidWheel(detIdStub);
      int ring = tTopo_->tidRing(detIdStub);
      int layer = tTopo_->getOTLayerNumber(detIdStub);
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
        CrackOverview->Fill(module, layer + 0.05 - (module % 2 * 0.1));

      if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TOB)) {  // Phase 2 Outer Tracker Barrel
        Stub_Barrel->Fill(layer);
        Stub_Barrel_XY->Fill(posStub.x(), posStub.y());
        Stub_Barrel_W->Fill(layer, rawBend - bendOffset);
        Stub_Barrel_O->Fill(layer, bendOffset);
      } else if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TID)) {  // Phase 2 Outer Tracker Endcap
        if (posStub.z() > 0) {
          Stub_Endcap_Fw_XY->Fill(posStub.x(), posStub.y());
        } else {
          Stub_Endcap_Bw_XY->Fill(posStub.x(), posStub.y());
        }
      }

      // Fill layer histograms
      for (enum Level fillingDepth = OT; fillingDepth <= LAYER; fillingDepth = Level(fillingDepth + 1)) {
        // Skip filling for barrel detIds on endcap-only depths
        if ((fillingDepth >= ENDCAP_SIDE && fillingDepth < LAYER) &&
            DetId(detIdStub).subdetId() == SiStripSubdetector::TOB)
          continue;
        std::string folderKey = phase2tkutil::getHistoId(detIdStub, tTopo_, 0, fillingDepth, false);
        auto layerMEiter = layerMEs_.find(folderKey);
        if (layerMEiter == layerMEs_.end())
          continue;
        TTStubMEs &local_mes = layerMEiter->second;

        local_mes.stubCounter++;

        if (detIdStub.subdetId() == static_cast<int>(StripSubdetector::TID)) {
          if (local_mes.NStubsByWheel)
            local_mes.NStubsByWheel->Fill(wheel);
          if (local_mes.NStubsByRing)
            local_mes.NStubsByRing->Fill(ring);
          if (local_mes.StubOffsetByWheel)
            local_mes.StubOffsetByWheel->Fill(wheel, bendOffset);
          if (local_mes.StubOffsetByRing)
            local_mes.StubOffsetByRing->Fill(ring, bendOffset);
          if (local_mes.StubWidthByWheel)
            local_mes.StubWidthByWheel->Fill(wheel, rawBend - bendOffset);
          if (local_mes.StubWidthByRing)
            local_mes.StubWidthByRing->Fill(ring, rawBend - bendOffset);
        }
      }  // end loop fillingDepth
    }  // end loop contentIter
  }  // end loop inputIter
  for (const auto &it : layerMEs_) {
    TTStubMEs local_mes = it.second;
    if (local_mes.NStubs)
      local_mes.NStubs->Fill(local_mes.stubCounter);
    local_mes.stubCounter = 0;
  }
}  // end of method

void Phase2OTMonitorTTStub::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &run, edm::EventSetup const &es) {
  using namespace phase2tkutil;

  // Whole OT Summaries
  iBooker.setCurrentFolder(topFolderName_);

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

  // Distributions
  Stub_Eta = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Eta"), iBooker);
  Stub_Phi = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Phi"), iBooker);
  Stub_R = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_R"), iBooker);
  Stub_bendFE = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_bendFE"), iBooker);
  Stub_bendBE = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_bendBE"), iBooker);
  Stub_isPS = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_isPS"), iBooker);

  // Positions
  iBooker.setCurrentFolder(topFolderName_ + "/Positions");
  Stub_Barrel_XY = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Global_Position_Barrel_XY"), iBooker);
  Stub_RZ = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Global_Position_RZ"), iBooker);
  Stub_Endcap_Fw_XY =
      book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Global_Position_Endcap_Fw_XY"), iBooker);
  Stub_Endcap_Bw_XY =
      book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Global_Position_Endcap_Bw_XY"), iBooker);

  // Barrel Summaries
  iBooker.setCurrentFolder(topFolderName_ + "/Barrel");
  Stub_Barrel = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("Num_L1Stubs_Barrel"), iBooker);
  Stub_Barrel_W = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Width_Barrel"), iBooker);
  Stub_Barrel_O = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Stub_Offset_Barrel"), iBooker);

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
      edm::LogInfo("Phase2OTMonitorTTStub")
          << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
          << "\t key=" << phase2tkutil::getHistoId(detId_raw, tTopo_, 0.0, 6, false) << std::endl;
      bookLayerHistos(iBooker, detId_raw, topFolderName_);
    }
  }
}

void Phase2OTMonitorTTStub::bookLayerHistos(DQMStore::IBooker &ibooker, uint32_t det_id, std::string &subdir) {
  for (enum Level bookingDepth = OT; bookingDepth <= LAYER; bookingDepth = Level(bookingDepth + 1)) {
    // Skip booking at endcap depths for barrel dets
    if ((bookingDepth >= ENDCAP_SIDE && bookingDepth < LAYER) &&
        DetId(det_id).subdetId() == static_cast<int>(StripSubdetector::TOB))
      continue;

    std::string folderName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, true);

    if (layerMEs_.find(folderName) == layerMEs_.end()) {
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + folderName);
      edm::LogInfo("Phase2OTMonitorTTStub") << " Booking Histograms in: " << subdir + "/" + folderName;
      TTStubMEs local_mes;

      local_mes.NStubs = phase2tkutil::book1DFromPSet(
          conf_.getParameter<edm::ParameterSet>("NStubsLayer"), ibooker, prettyName, bookingDepth);
      if (DetId(det_id).subdetId() == static_cast<int>(StripSubdetector::TID)) {
        if (bookingDepth >= SUBSTRUCTURE && bookingDepth < LAYER) {
          if (bookingDepth != ENDCAP_WHEEL) {
            local_mes.NStubsByWheel = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NStubsByWheel"), ibooker, prettyName);
            local_mes.StubOffsetByWheel = phase2tkutil::book2DFromPSet(
                conf_.getParameter<edm::ParameterSet>("StubOffsetByWheel"), ibooker, prettyName);
            local_mes.StubWidthByWheel = phase2tkutil::book2DFromPSet(
                conf_.getParameter<edm::ParameterSet>("StubWidthByWheel"), ibooker, prettyName);
          }
          if (bookingDepth != ENDCAP_RING) {
            local_mes.NStubsByRing = phase2tkutil::book1DFromPSet(
                conf_.getParameter<edm::ParameterSet>("NStubsByRing"), ibooker, prettyName);
            local_mes.StubOffsetByRing = phase2tkutil::book2DFromPSet(
                conf_.getParameter<edm::ParameterSet>("StubOffsetByRing"), ibooker, prettyName);
            local_mes.StubWidthByRing = phase2tkutil::book2DFromPSet(
                conf_.getParameter<edm::ParameterSet>("StubWidthByRing"), ibooker, prettyName);
          }
        }
      }
      layerMEs_.emplace(folderName, local_mes);
    }
  }
}

void Phase2OTMonitorTTStub::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // CRACK
  phase2tkutil::add2DDesc(desc,
                          "CrackOverview",
                          "Crack_Overview_L1Stubs",
                          "Crack_Overview_stubs",
                          "Module",
                          "Layer",
                          13.0,
                          0.0,
                          13,
                          0.0,
                          7.5,
                          13);

  // Position
  phase2tkutil::add2DDesc(desc,
                          "L1Stub_Global_Position_Barrel_XY",
                          "L1Stub_Global_Position_Barrel_XY",
                          "L1Stub_Global_Position_Barrel_XY",
                          "L1 Stub Barrel position x [cm]",
                          "L1 Stub Barrel position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Stub_Global_Position_Endcap_Fw_XY",
                          "L1Stub_Global_Position_Endcap_Fw_XY",
                          "L1Stub_Global_Position_Endcap_Fw_XY",
                          "L1 Stub Endcap position x [cm]",
                          "L1 Stub Endcap position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Stub_Global_Position_Endcap_Bw_XY",
                          "L1Stub_Global_Position_Endcap_Bw_XY",
                          "L1Stub_Global_Position_Endcap_Bw_XY",
                          "L1 Stub Endcap position x [cm]",
                          "L1 Stub Endcap position y [cm]",
                          960,
                          -120,
                          120,
                          960,
                          -120,
                          120);
  phase2tkutil::add2DDesc(desc,
                          "L1Stub_Global_Position_RZ",
                          "L1Stub_Global_Position_RZ",
                          "L1Stub_Global_Position_RZ",
                          "L1 Stub position z [cm]",
                          "L1 Stub position #rho [cm]",
                          900,
                          -300,
                          300,
                          900,
                          0,
                          120);

  // Stub distributions
  phase2tkutil::add1DDesc(desc, "L1Stub_Eta", "L1Stub_Eta", "L1Stub_Eta", "#eta", "# L1 Stubs", 45, -5, 5);
  phase2tkutil::add1DDesc(desc, "L1Stub_Phi", "L1Stub_Phi", "L1Stub_Phi", "#phi", "# L1 Stubs", 60, -3.5, 3.5);
  phase2tkutil::add1DDesc(desc, "L1Stub_R", "L1Stub_R", "L1Stub_R", "R", "# L1 Stubs", 45, 0, 120);
  phase2tkutil::add1DDesc(
      desc, "L1Stub_bendFE", "L1Stub_bendFE", "L1Stub_bendFE", "Trigger bend", "# L1 Stubs", 69, -8.625, 8.625);
  phase2tkutil::add1DDesc(
      desc, "L1Stub_bendBE", "L1Stub_bendBE", "L1Stub_bendBE", "Hardware bend", "# L1 Stubs", 69, -8.625, 8.625);
  phase2tkutil::add1DDesc(desc, "L1Stub_isPS", "L1Stub_isPS", "L1Stub_isPS", "Is PS?", "# L1 Stubs", 2, 0, 2);

  // Barrel Histos
  phase2tkutil::add1DDesc(
      desc, "Num_L1Stubs_Barrel", "Num_L1Stubs_Barrel", "Num_L1Stubs_Barrel", "Barrel Layer", "# L1 Stubs", 6, 0.5, 6.5);
  phase2tkutil::add2DDesc(desc,
                          "L1Stub_Width_Barrel",
                          "L1Stub_Width_Barrel",
                          "L1Stub_Width_Barrel",
                          "Barrel Layer",
                          "Displacement - Offset",
                          6,
                          0.5,
                          6.5,
                          43,
                          -10.75,
                          10.75);
  phase2tkutil::add2DDesc(desc,
                          "L1Stub_Offset_Barrel",
                          "L1Stub_Offset_Barrel",
                          "L1Stub_Offset_Barrel",
                          "Barrel Layer",
                          "Trigger Offset",
                          6,
                          0.5,
                          6.5,
                          43,
                          -10.75,
                          10.75);

  // Layer histos
  phase2tkutil::add1DDesc(desc,
                          "NStubsLayer",
                          "Num_L1Stubs_Per_Event",
                          "Number of L1Stubs in {} per event",
                          "Number of stubs",
                          "Number of events",
                          100,
                          0,
                          300000);
  phase2tkutil::add1DDesc(desc,
                          "NStubsByWheel",
                          "Num_L1Stubs_Wheels",
                          "Number of L1Stubs in {} by wheel",
                          "Wheel",
                          "Number of stubs",
                          6,
                          0.5,
                          6.5);
  phase2tkutil::add1DDesc(desc,
                          "NStubsByRing",
                          "Num_L1Stubs_Rings",
                          "Number of L1Stubs in {} by ring",
                          "Ring",
                          "Number of stubs",
                          16,
                          0.5,
                          16.5);

  phase2tkutil::add2DDesc(desc,
                          "StubWidthByRing",
                          "L1Stub_Width_By_Ring",
                          "L1Stub width in {} by ring",
                          "Ring",
                          "Displacement - Offset",
                          16,
                          0.5,
                          16.5,
                          43,
                          -10.75,
                          10.75);
  phase2tkutil::add2DDesc(desc,
                          "StubWidthByWheel",
                          "L1Stub_Width_By_Wheel",
                          "L1Stub width in {} by wheel",
                          "Wheel",
                          "Displacement - Offset",
                          5,
                          0.5,
                          5.5,
                          43,
                          -10.75,
                          10.75);

  phase2tkutil::add2DDesc(desc,
                          "StubOffsetByRing",
                          "L1Stub_Offset_By_Ring",
                          "L1Stub offset in {} by ring",
                          "Ring",
                          "Trigger Offset",
                          16,
                          0.5,
                          16.5,
                          43,
                          -10.75,
                          10.75);
  phase2tkutil::add2DDesc(desc,
                          "StubOffsetByWheel",
                          "L1Stub_Offset_By_Wheel",
                          "L1Stub offset in {} by wheel",
                          "Wheel",
                          "Trigger Offset",
                          5,
                          0.5,
                          5.5,
                          43,
                          -10.75,
                          10.75);

  desc.add<std::string>("TopFolderName", "OuterTracker");
  desc.add<edm::InputTag>("TTStubs", edm::InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"));
  descriptions.add("Phase2OTMonitorTTStub", desc);
}

DEFINE_FWK_MODULE(Phase2OTMonitorTTStub);
