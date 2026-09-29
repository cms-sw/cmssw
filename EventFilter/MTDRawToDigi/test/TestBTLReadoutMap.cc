#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"

#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "CondFormats/MTDObjects/interface/BTLReadoutMap.h"
#include "CondFormats/DataRecord/interface/BTLReadoutMapRcd.h"
#include "EventFilter/MTDRawToDigi/interface/BTLElectronicsMapping.h"

#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/Math/interface/Rounding.h"
#include <DD4hep/DD4hepUnits.h>

using namespace cms;

class TestBTLReadoutMap : public edm::one::EDAnalyzer<> {
public:
  explicit TestBTLReadoutMap(const edm::ParameterSet&);
  ~TestBTLReadoutMap() override = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(cms::DDFilteredView& fv);

private:
  const edm::ESInputTag tag_;
  std::string ddTopNodeName_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;

  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
  edm::ESGetToken<BTLReadoutMap, BTLReadoutMapRcd> readoutMapToken_;
};

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using angle_units::operators::convertRadToDeg;
using cms_rounding::roundIfNear0;

TestBTLReadoutMap::TestBTLReadoutMap(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      thisN_(),
      btlNS_() {
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
  readoutMapToken_ = esConsumes<BTLReadoutMap, BTLReadoutMapRcd>();
}

void TestBTLReadoutMap::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();
  auto btlCrysLayout = MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode());

  auto const& btlReadoutMap = iSetup.getData(readoutMapToken_);
  if (btlReadoutMap.size() == 0) {
    edm::LogError("TestBTLReadoutMap") << "BTL readout map is empty !";
  }

  if (ddTopNodeName_ != "BarrelTimingLayer") {
    edm::LogWarning("TestBTLReadoutMap") << ddTopNodeName_ << "Not valid top BarrelTimingLayer volume";
    return;
  }

  DDFilteredView fv(pDD.product(), pDD.product()->description()->worldVolume());
  fv.next(0);

  DDSpecParRefs specs;
  pSP.product()->filter(specs, "ReadOutName", "FastTimerHitsBarrel");

  bool insideBTL = false;
  uint32_t startLevel = 0;

  do {
    if (dd4hep::dd::noNamespace(fv.name()) == "BarrelTimingLayer") {
      insideBTL = true;
      startLevel = fv.navPos().size();
      edm::LogInfo("TestBTLReadoutMap") << "insideBTL = " << insideBTL;
      if (static_cast<int>(btlCrysLayout) < static_cast<int>(BTLDetId::CrysLayout::v4)) {
        edm::LogInfo("TestBTLReadoutMap")
            << "BTL electronics mapping not available for BTL crystal layout " << static_cast<int>(btlCrysLayout)
            << ", use layout 7 (v4) or later!" << std::endl;
      }
      continue;
    }

    if (insideBTL && fv.navPos().size() < startLevel) {
      break;  // exited BTL --> break loop
    }

    if (!insideBTL)
      continue;

    std::stringstream ss;

    theBaseNumber(fv);

    bool isSens = false;

    for (auto const& t : specs) {
      for (auto const& it : t.second->paths) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(fv.name()), dd4hep::dd::realTopName(it))) {
          isSens = true;
          break;
        }
      }
    }

    if (isSens) {
      std::stringstream sunitt;
      std::stringstream snum;

      BTLDetId theId(btlNS_.getUnitID(thisN_));
      sunitt << theId.rawId();
      snum << theId;

      // --- Compute the crystal ends positions
      dd4hep::Box mySens(fv.solid());
      DD3Vector x, y, z;
      fv.rotation().GetComponents(x, y, z);
      DD3Vector zeroLocal(0., 0., 0.);
      DD3Vector plusLocal(mySens.x(), 0., 0.);
      DD3Vector plusGlobal = (fv.rotation())(plusLocal) + fv.translation();
      DD3Vector minusLocal(-mySens.x(), 0., 0.);
      DD3Vector minusGlobal = (fv.rotation())(minusLocal) + fv.translation();

      if (static_cast<int>(btlCrysLayout) >= static_cast<int>(BTLDetId::CrysLayout::v4)) {
        auto fround = [&](double in) {
          std::stringstream ss;
          ss << std::fixed << std::setw(14) << roundIfNear0(in);
          return ss.str();
        };

        snum << "\n";
        snum << "- location = " << fround(minusGlobal.X() / dd4hep::mm) << fround(minusGlobal.Y() / dd4hep::mm)
             << fround(minusGlobal.Z() / dd4hep::mm) << " r = " << fround(minusGlobal.Rho() / dd4hep::mm)
             << " phi = " << fround(convertRadToDeg(minusGlobal.Phi())) << "\n";
        snum << "+ location = " << fround(plusGlobal.X() / dd4hep::mm) << fround(plusGlobal.Y() / dd4hep::mm)
             << fround(plusGlobal.Z() / dd4hep::mm) << " r = " << fround(plusGlobal.Rho() / dd4hep::mm)
             << " phi = " << fround(convertRadToDeg(plusGlobal.Phi())) << "\n";

        // --- Check electronics ids
        auto const& elecIds = btlReadoutMap.getElectronicsId(theId);

        snum << "\n";
        snum << "BTLElectronicsId (minus) : " << elecIds.minus << "\n"
             << "BTLElectronicsId (plus)  : " << elecIds.plus << "\n"
             << "\n";

        auto const& detMinus = btlReadoutMap.getDetId(elecIds.minus);
        auto const& detPlus = btlReadoutMap.getDetId(elecIds.plus);

        if (detMinus.rawId() != theId.rawId()) {
          edm::LogError("TestBTLReadoutMap") << "Reverse mapping mismatch for minus side!";
        }
        if (detPlus.rawId() != theId.rawId()) {
          edm::LogError("TestBTLReadoutMap") << "Reverse mapping mismatch for plus side!";
        }

        BTLElectronicsMapping btlElMapping = BTLElectronicsMapping();
        snum << " TOFHIRASIC: " << btlElMapping.TOFHIRASIC(theId)
             << "\n SiPMCh minus: " << btlElMapping.SiPMCh(theId, 0) << " plus: " << btlElMapping.SiPMCh(theId, 1)
             << "\n TOFHIRCh minus: " << btlElMapping.TOFHIRCh(theId, 0) << " plus: " << btlElMapping.TOFHIRCh(theId, 1)
             << "\n";
        snum << "\n";

        snum << "----------------------------------------------------------------------------" << std::endl;
        snum << " DM, SM, chipId: " << theId.dmodule() << ", " << theId.smodule() << ", "
             << btlElMapping.TOFHIRASIC(theId) << "  --> e-link: " << elecIds.minus.eLinkId() << " ("
             << elecIds.plus.eLinkId() << ")\n"
             << " Side, Tray, RU: " << theId.mtdSide() << ", " << theId.mtdRR() << ", " << theId.runit()
             << "  --> HS-link : " << elecIds.minus.hsLinkId() << " (" << elecIds.plus.hsLinkId()
             << ")   FED ID / S-link : " << elecIds.minus.fedId() << " (" << elecIds.plus.fedId() << ")\n"

             << " BTLElectronicsId (minus) rawId : " << elecIds.minus.rawId()
             << " --> Side, Tray, RU, DM, SM: " << detMinus.mtdSide() << ", " << detMinus.mtdRR() << ", "
             << detMinus.runit() << ", " << detMinus.dmodule() << ", " << detMinus.smodule() << "\n"

             << " BTLElectronicsId (plus)  rawId : " << elecIds.plus.rawId()
             << " --> Side, Tray, RU, DM, SM: " << detPlus.mtdSide() << ", " << detPlus.mtdRR() << ", "
             << detPlus.runit() << ", " << detPlus.dmodule() << ", " << detPlus.smodule() << "\n";
        snum << "----------------------------------------------------------------------------" << std::endl;
      }
      edm::LogInfo("TestBTLReadoutMap") << snum.str();
    }
  } while (fv.next(0));
}

void TestBTLReadoutMap::theBaseNumber(cms::DDFilteredView& fv) {
  thisN_.reset();
  thisN_.setSize(fv.navPos().size());

  for (uint ii = 0; ii < fv.navPos().size(); ii++) {
    std::string_view name((fv.geoHistory()[ii])->GetName());
    size_t ipos = name.rfind('_');
    thisN_.addLevel(name.substr(0, ipos), fv.copyNos()[ii]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("TestBTLReadoutMap") << ii << " " << name.substr(0, ipos) << " " << fv.copyNos()[ii];
#endif
  }
}

DEFINE_FWK_MODULE(TestBTLReadoutMap);
