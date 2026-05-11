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
//#include "Geometry/MTDCommonData/interface/BTLElectronicsMapping.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "CondFormats/MTDObjects/interface/BTLElectronicsMapping.h"

#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/Math/interface/Rounding.h"
#include <DD4hep/DD4hepUnits.h>

using namespace cms;

class TestBTLElectronicsMapping : public edm::one::EDAnalyzer<> {
public:
  explicit TestBTLElectronicsMapping(const edm::ParameterSet&);
  ~TestBTLElectronicsMapping() override = default;

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
};

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using angle_units::operators::convertRadToDeg;
using cms_rounding::roundIfNear0;

TestBTLElectronicsMapping::TestBTLElectronicsMapping(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      thisN_(),
      btlNS_() {
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
}

void TestBTLElectronicsMapping::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();
  auto btlCrysLayout = MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode());

  if (ddTopNodeName_ != "BarrelTimingLayer") {
    edm::LogWarning("TestBTLElectronicsMapping") << ddTopNodeName_ << "Not valid top BarrelTimingLayer volume";
    return;
  }

  DDFilteredView fv(pDD.product(), pDD.product()->description()->worldVolume());
  fv.next(0);
  edm::LogInfo("TestBTLElectronicsMapping") << fv.name();

  DDSpecParRefs specs;
  pSP.product()->filter(specs, "ReadOutName", "FastTimerHitsBarrel");

  bool insideBTL = false;
  uint32_t startLevel = 0;

  do {
    if (dd4hep::dd::noNamespace(fv.name()) == "BarrelTimingLayer") {
      insideBTL = true;
      startLevel = fv.navPos().size();
      edm::LogInfo("TestBTLElectronicsMapping") << "insideBTL = " << insideBTL;
      if (static_cast<int>(btlCrysLayout) < static_cast<int>(BTLDetId::CrysLayout::v4)) {
        edm::LogInfo("DD4hep_TestMTDIdealGeometry")
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

      // Compute the crystal ends positions
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

        BTLElectronicsMapping btlElMap = BTLElectronicsMapping(btlCrysLayout);
        snum << "\n";
        snum << "----------------------------------------------------------------------------" << std::endl;
        snum << " CCBoard: " << btlElMap.CCBoard(theId) << " FEBoard: " << btlElMap.FEBoard(theId)
             << " TOFHIRASIC: " << btlElMap.TOFHIRASIC(theId) << "\n SiPMCh   minus: " << btlElMap.SiPMCh(theId, 0)
             << " minus: " << btlElMap.SiPMCh(theId, 1) << "\n TOFHIRCh minus: " << btlElMap.TOFHIRCh(theId, 0)
             << " minus: " << btlElMap.TOFHIRCh(theId, 1) << "\n";
        snum << "\n";
        snum << " DM, SM, chipId    : " << theId.dmodule() << ", " << theId.smodule() << ", "
             << btlElMap.TOFHIRASIC(theId) << "  e-link: " << btlElMap.elink(theId) << "\n"
             << " DM, SM from e-link: " << btlElMap.elinkToSM(btlElMap.elink(theId)).first << ", "
             << btlElMap.elinkToSM(btlElMap.elink(theId)).second << "\n";
        snum << "\n";
        snum << " Tray: " << theId.mtdRR() << "  RU: " << theId.runit() << "  hs-link : " << btlElMap.hslink(theId)
             << "\n"
             << " RU from hs-link: " << btlElMap.hslinkToRU(btlElMap.hslink(theId)) << "\n";
        snum << "\n";
        snum << " Tray, side                     : " << theId.mtdRR() << ", " << theId.mtdSide()
             << "   S-link : " << btlElMap.Slink(theId) << "\n"
             << " Tray, side from S-link, HS-link: "
             << btlElMap.getTrayFromLinks(btlElMap.Slink(theId), btlElMap.hslink(theId)).first << ", "
             << btlElMap.getTrayFromLinks(btlElMap.Slink(theId), btlElMap.hslink(theId)).second << "\n";
        snum << "----------------------------------------------------------------------------" << std::endl;
      }
      edm::LogInfo("TestBTLElectronicsMapping") << snum.str();
    }
  } while (fv.next(0));
}

void TestBTLElectronicsMapping::theBaseNumber(cms::DDFilteredView& fv) {
  thisN_.reset();
  thisN_.setSize(fv.navPos().size());

  for (uint ii = 0; ii < fv.navPos().size(); ii++) {
    std::string_view name((fv.geoHistory()[ii])->GetName());
    size_t ipos = name.rfind('_');
    thisN_.addLevel(name.substr(0, ipos), fv.copyNos()[ii]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("TestBTLElectronicsMapping") << ii << " " << name.substr(0, ipos) << " " << fv.copyNos()[ii];
#endif
  }
}

DEFINE_FWK_MODULE(TestBTLElectronicsMapping);
