// -*- C++ -*-
//
// Package:    TrackerTreeGenerator
// Class:      TrackerTreeGenerator
//
/**\class TrackerTreeGenerator TrackerTreeGenerator.cc Alignment/TrackerAlignment/plugins/TrackerTreeGenerator.cc
 Description: <one line class summary>
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Johannes Hauk
//         Created:  Fri Jan 16 14:09:52 CET 2009
//         Modified by: Gregor Mittag (DESY)
//
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "Alignment/TrackerAlignment/interface/TrackerTreeVariables.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "TTree.h"
//
// class decleration
//

class TrackerTreeGenerator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackerTreeGenerator(const edm::ParameterSet&);
  ~TrackerTreeGenerator() override = default;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<PTrackerAdditionalParametersPerDet, PTrackerAdditionalParametersPerDetRcd> ptitpToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  const bool createEntryForDoubleSidedModule_;
  std::vector<TrackerTreeVariables> vTkTreeVar_;
  edm::ParameterSet config_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackerTreeGenerator::TrackerTreeGenerator(const edm::ParameterSet& config)
    : geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      ptitpToken_(esConsumes()),
      topoToken_(esConsumes()),
      createEntryForDoubleSidedModule_(config.getParameter<bool>("createEntryForDoubleSidedModule")),
      config_(config) {
  usesResource(TFileService::kSharedResource);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TrackerTreeGenerator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // now try to take directly the ideal geometry independent of used geometry in Global Tag
  const GeometricDet* geometricDet = &iSetup.getData(geomDetToken_);
  const PTrackerParameters& ptp = iSetup.getData(ptpToken_);
  const PTrackerAdditionalParametersPerDet* ptitp = &iSetup.getData(ptitpToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoToken_);

  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  const TrackerGeometry* tkGeom = trackerBuilder.build(geometricDet, ptitp, ptp, tTopo);
  AlignableTracker alignableTracker{tkGeom, tTopo};
  const auto& ns = alignableTracker.trackerNameSpace();

  edm::LogInfo("TrackerTreeGenerator") << "@SUB=TrackerTreeGenerator::analyze"
                                       << "There are " << tkGeom->detIds().size() << " dets and "
                                       << tkGeom->detUnitIds().size() << " detUnits in the Geometry Record";

  if (createEntryForDoubleSidedModule_) {
    edm::LogInfo("TrackerTreeGenerator") << "@SUB=TrackerTreeGenerator::analyze"
                                         << "Create entry for each module AND one entry for virtual "
                                         << "double-sided module in addition";
  } else {
    edm::LogInfo("TrackerTreeGenerator") << "@SUB=TrackerTreeGenerator::analyze"
                                         << "Create one entry for each physical module, do NOT create additional "
                                         << "entry for virtual double-sided module";
  }

  for (const auto& detId : tkGeom->detIds()) {
    const GeomDet& geomDet = *tkGeom->idToDet(detId);
    const Surface& surface = geomDet.surface();

    TrackerTreeVariables tkTreeVar;
    const auto rawId = detId.rawId();
    tkTreeVar.rawId = rawId;
    tkTreeVar.subdetId = detId.subdetId();

    switch (tkTreeVar.subdetId) {
      case PixelSubdetector::PixelBarrel:
        tkTreeVar.layer = tTopo->pxbLayer(detId);
        tkTreeVar.half = ns.tpb().halfBarrelNumber(rawId);
        tkTreeVar.rod = tTopo->pxbLadder(detId);  // ... so, ladder is not per halfBarrel-Layer, but per barrel-layer!
        tkTreeVar.module = tTopo->pxbModule(detId);
        break;
      case PixelSubdetector::PixelEndcap:
        tkTreeVar.layer = tTopo->pxfDisk(detId);
        tkTreeVar.side = tTopo->pxfSide(detId);
        tkTreeVar.half = ns.tpe().halfCylinderNumber(rawId);
        tkTreeVar.blade = tTopo->pxfBlade(detId);
        tkTreeVar.panel = tTopo->pxfPanel(detId);
        tkTreeVar.module = tTopo->pxfModule(detId);
        break;
      case StripSubdetector::TIB:
        tkTreeVar.layer = tTopo->tibLayer(detId);
        tkTreeVar.side = tTopo->tibStringInfo(detId)[0];
        tkTreeVar.half = ns.tib().halfShellNumber(rawId);
        tkTreeVar.rod = tTopo->tibStringInfo(detId)[2];
        tkTreeVar.outerInner = tTopo->tibStringInfo(detId)[1];
        tkTreeVar.module = tTopo->tibModule(detId);
        tkTreeVar.isDoubleSide = tTopo->tibIsDoubleSide(detId);
        tkTreeVar.isRPhi = tTopo->tibIsRPhi(detId);
        tkTreeVar.isStereo = tTopo->tibIsStereo(detId);
        break;
      case StripSubdetector::TID:
        tkTreeVar.layer = tTopo->tidWheel(detId);
        tkTreeVar.side = tTopo->tidSide(detId);
        tkTreeVar.ring = tTopo->tidRing(detId);
        tkTreeVar.outerInner = tTopo->tidModuleInfo(detId)[0];
        tkTreeVar.module = tTopo->tidModuleInfo(detId)[1];
        tkTreeVar.isDoubleSide = tTopo->tidIsDoubleSide(detId);
        tkTreeVar.isRPhi = tTopo->tidIsRPhi(detId);
        tkTreeVar.isStereo = tTopo->tidIsStereo(detId);
        break;
      case StripSubdetector::TOB:
        tkTreeVar.layer = tTopo->tobLayer(detId);
        tkTreeVar.side = tTopo->tobRodInfo(detId)[0];
        tkTreeVar.rod = tTopo->tobRodInfo(detId)[1];
        tkTreeVar.module = tTopo->tobModule(detId);
        tkTreeVar.isDoubleSide = tTopo->tobIsDoubleSide(detId);
        tkTreeVar.isRPhi = tTopo->tobIsRPhi(detId);
        tkTreeVar.isStereo = tTopo->tobIsStereo(detId);
        break;
      case StripSubdetector::TEC:
        tkTreeVar.layer = tTopo->tecWheel(detId);
        tkTreeVar.side = tTopo->tecSide(detId);
        tkTreeVar.ring = tTopo->tecRing(detId);
        tkTreeVar.petal = tTopo->tecPetalInfo(detId)[1];
        tkTreeVar.outerInner = tTopo->tecPetalInfo(detId)[0];
        tkTreeVar.module = tTopo->tecModule(detId);
        tkTreeVar.isDoubleSide = tTopo->tecIsDoubleSide(detId);
        tkTreeVar.isRPhi = tTopo->tecIsRPhi(detId);
        tkTreeVar.isStereo = tTopo->tecIsStereo(detId);
        break;
    }

    LocalPoint lPModule(0., 0., 0.), lUDirection(1., 0., 0.), lVDirection(0., 1., 0.), lWDirection(0., 0., 1.);
    GlobalPoint gPModule = surface.toGlobal(lPModule), gUDirection = surface.toGlobal(lUDirection),
                gVDirection = surface.toGlobal(lVDirection), gWDirection = surface.toGlobal(lWDirection);
    double dR(999.), dPhi(999.), dZ(999.);
    switch (tkTreeVar.subdetId) {
      case PixelSubdetector::PixelBarrel:
      case StripSubdetector::TIB:
      case StripSubdetector::TOB:
        dR = gWDirection.perp() - gPModule.perp();
        dPhi = deltaPhi(gUDirection.barePhi(), gPModule.barePhi());
        dZ = gVDirection.z() - gPModule.z();
        tkTreeVar.uDirection = dPhi > 0. ? 1 : -1;
        tkTreeVar.vDirection = dZ > 0. ? 1 : -1;
        tkTreeVar.wDirection = dR > 0. ? 1 : -1;
        break;
      case PixelSubdetector::PixelEndcap:
        dR = gUDirection.perp() - gPModule.perp();
        dPhi = deltaPhi(gVDirection.barePhi(), gPModule.barePhi());
        dZ = gWDirection.z() - gPModule.z();
        tkTreeVar.uDirection = dR > 0. ? 1 : -1;
        tkTreeVar.vDirection = dPhi > 0. ? 1 : -1;
        tkTreeVar.wDirection = dZ > 0. ? 1 : -1;
        break;
      case StripSubdetector::TID:
      case StripSubdetector::TEC:
        dR = gVDirection.perp() - gPModule.perp();
        dPhi = deltaPhi(gUDirection.barePhi(), gPModule.barePhi());
        dZ = gWDirection.z() - gPModule.z();
        tkTreeVar.uDirection = dPhi > 0. ? 1 : -1;
        tkTreeVar.vDirection = dR > 0. ? 1 : -1;
        tkTreeVar.wDirection = dZ > 0. ? 1 : -1;
        break;
    }
    tkTreeVar.posR = gPModule.perp();
    tkTreeVar.posPhi = gPModule.barePhi();  // = gPModule.barePhi().degrees();
    tkTreeVar.posEta = gPModule.eta();
    tkTreeVar.posX = gPModule.x();
    tkTreeVar.posY = gPModule.y();
    tkTreeVar.posZ = gPModule.z();

    if (auto stripGeomDetUnit = dynamic_cast<const StripGeomDetUnit*>(&geomDet)) {  //is it a single physical module?
      switch (tkTreeVar.subdetId) {
        case StripSubdetector::TIB:
        case StripSubdetector::TOB:
        case StripSubdetector::TID:
        case StripSubdetector::TEC:
          auto& topol = dynamic_cast<const StripTopology&>(stripGeomDetUnit->specificTopology());
          tkTreeVar.nStrips = topol.nstrips();
          break;
      }
    }

    if (!createEntryForDoubleSidedModule_) {
      // do so only for individual modules and not also one entry for the combined doubleSided Module
      if (tkTreeVar.isDoubleSide)
        continue;
    }
    vTkTreeVar_.push_back(tkTreeVar);
  }
}

// ------------ method called once each job just before starting event loop  ------------
void TrackerTreeGenerator::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TrackerTreeGenerator::endJob() {
  UInt_t rawId(999), subdetId(999), layer(999), side(999), half(999), rod(999), ring(999), petal(999), blade(999),
      panel(999), outerInner(999), module(999), nStrips(999);
  Bool_t isDoubleSide(false), isRPhi(false), isStereo(false);
  Int_t uDirection(999), vDirection(999), wDirection(999);
  Float_t posR(999.F), posPhi(999.F), posEta(999.F), posX(999.F), posY(999.F), posZ(999.F);

  edm::Service<TFileService> fileService;
  TFileDirectory treeDir = fileService->mkdir("TrackerTree");
  auto trackerTree{treeDir.make<TTree>("TrackerTree", "IDs of all modules (ideal geometry)")};
  trackerTree->Branch("RawId", &rawId, "RawId/i");
  trackerTree->Branch("SubdetId", &subdetId, "SubdetId/i");
  trackerTree->Branch("Layer", &layer, "Layer/i");                 // Barrel: Layer, Forward: Disk
  trackerTree->Branch("Side", &side, "Side/i");                    // Rod/Ring in +z or -z
  trackerTree->Branch("Half", &half, "Half/i");                    // PXB: HalfBarrel, PXF: HalfCylinder, TIB: HalfShell
  trackerTree->Branch("Rod", &rod, "Rod/i");                       // Barrel (Ladder or String or Rod)
  trackerTree->Branch("Ring", &ring, "Ring/i");                    // Forward
  trackerTree->Branch("Petal", &petal, "Petal/i");                 // TEC
  trackerTree->Branch("Blade", &blade, "Blade/i");                 // PXF
  trackerTree->Branch("Panel", &panel, "Panel/i");                 // PXF
  trackerTree->Branch("OuterInner", &outerInner, "OuterInner/i");  // front/back String,Ring,Petal
  trackerTree->Branch("Module", &module, "Module/i");              // Module ID
  trackerTree->Branch("NStrips", &nStrips, "NStrips/i");
  trackerTree->Branch("IsDoubleSide", &isDoubleSide, "IsDoubleSide/O");
  trackerTree->Branch("IsRPhi", &isRPhi, "IsRPhi/O");
  trackerTree->Branch("IsStereo", &isStereo, "IsStereo/O");
  trackerTree->Branch("UDirection", &uDirection, "UDirection/I");
  trackerTree->Branch("VDirection", &vDirection, "VDirection/I");
  trackerTree->Branch("WDirection", &wDirection, "WDirection/I");
  trackerTree->Branch("PosR", &posR, "PosR/F");
  trackerTree->Branch("PosPhi", &posPhi, "PosPhi/F");
  trackerTree->Branch("PosEta", &posEta, "PosEta/F");
  trackerTree->Branch("PosX", &posX, "PosX/F");
  trackerTree->Branch("PosY", &posY, "PosY/F");
  trackerTree->Branch("PosZ", &posZ, "PosZ/F");

  for (const auto& iTree : vTkTreeVar_) {
    rawId = iTree.rawId;
    subdetId = iTree.subdetId;
    layer = iTree.layer;
    side = iTree.side;
    half = iTree.half;
    rod = iTree.rod;
    ring = iTree.ring;
    petal = iTree.petal;
    blade = iTree.blade;
    panel = iTree.panel;
    outerInner = iTree.outerInner;
    module = iTree.module;
    nStrips = iTree.nStrips;
    isDoubleSide = iTree.isDoubleSide;
    isRPhi = iTree.isRPhi;
    isStereo = iTree.isStereo;
    uDirection = iTree.uDirection;
    vDirection = iTree.vDirection;
    wDirection = iTree.wDirection;
    posR = iTree.posR;
    posPhi = iTree.posPhi;
    posEta = iTree.posEta;
    posX = iTree.posX;
    posY = iTree.posY;
    posZ = iTree.posZ;

    trackerTree->Fill();
  }
  edm::LogInfo("TrackerTreeGenerator") << "@SUB=TrackerTreeGenerator::endJob"
                                       << "TrackerTree contains " << vTkTreeVar_.size() << " entries overall";
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerTreeGenerator);
