//#define EDM_ML_DEBUG

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

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"

#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Rounding.h"
#include <DD4hep/DD4hepUnits.h>

using namespace cms;
using namespace geant_units::operators;

class DD4hep_TestBTLPixelTopology : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestBTLPixelTopology(const edm::ParameterSet&);
  ~DD4hep_TestBTLPixelTopology() = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(cms::DDFilteredView& fv);

private:
  const edm::ESInputTag tag_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;

  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;

  std::stringstream sunitt;
  constexpr static double tolerance{0.5e-3_mm};
};

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using cms_rounding::roundIfNear0;

DD4hep_TestBTLPixelTopology::DD4hep_TestBTLPixelTopology(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")), thisN_(), btlNS_() {
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>(tag_);
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>(tag_);
}

void DD4hep_TestBTLPixelTopology::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

  if (!pDD.isValid()) {
    edm::LogError("DD4hep_TestBTLPixelTopology") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("DD4hep_TestBTLPixelTopology") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("DD4hep_TestBTLPixelTopology") << "NO label found pDD.description() returned false.";
  }

  if (!pSP.isValid()) {
    edm::LogError("DD4hep_TestBTLPixelTopology") << "ESTransientHandle<DDSpecParRegistry> pSP is not valid!";
    return;
  }

  auto pTP = iSetup.getTransientHandle(mtdtopoToken_);
  if (!pTP.isValid()) {
    edm::LogError("DD4hep_TestBTLPixelTopology") << "ESTransientHandle<MTDTopology> pTP is not valid!";
    return;
  } else {
    edm::LogInfo("DD4hep_TestBTLPixelTopology")
        << "MTD topology mode = " << pTP.product()->getMTDTopologyMode() << " BTLDetId:CrysLayout = "
        << static_cast<int>(MTDTopologyMode::crysLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()));
  }

  auto pDG = iSetup.getTransientHandle(mtdgeoToken_);
  if (!pDG.isValid()) {
    edm::LogError("DD4hep_TestBTLPixelTopology") << "ESTransientHandle<MTDGeometry> pDG is not valid!";
    return;
  } else {
    edm::LogInfo("DD4hep_TestBTLPixelTopology")
        << "Geometry node for MTDGeom is  " << &(*pDG) << "\n"
        << " # detectors = " << pDG.product()->detUnits().size() << "\n"
        << " # types     = " << pDG.product()->detTypes().size() << "\n"
        << " # BTL dets  = " << pDG.product()->detsBTL().size() << "\n"
        << " # ETL dets  = " << pDG.product()->detsETL().size() << "\n"
        << " # layers " << pDG.product()->geomDetSubDetector(1) << "  = " << pDG.product()->numberOfLayers(1) << "\n"
        << " # layers " << pDG.product()->geomDetSubDetector(2) << "  = " << pDG.product()->numberOfLayers(2) << "\n";
  }

  DDFilteredView fv(pDD.product(), pDD.product()->description()->worldVolume());
  fv.next(0);

  DDSpecParRefs specs;
  std::string attribute("ReadOutName"), name("FastTimerHitsBarrel");
  pSP.product()->filter(specs, attribute, name);

  edm::LogVerbatim("DD4hep_TestBTLPixelTopology").log([&specs](auto& log) {
    log << "Filtered DD SpecPar Registry size: " << specs.size() << "\n";
    for (const auto& t : specs) {
      log << "\nSpecPar " << t.first << ":\nRegExps { ";
      for (const auto& ki : t.second->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t.second->spars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << " ";
        }
        log << "\n ";
      }
    }
  });

  bool isBarrel = true;
  bool exitLoop = false;
  uint32_t level(0);

  do {
    if (dd4hep::dd::noNamespace(fv.name()) == "BarrelTimingLayer") {
      isBarrel = true;
      edm::LogInfo("DD4hep_TestBTLPixelTopology") << "isBarrel = " << isBarrel;
    } else if (dd4hep::dd::noNamespace(fv.name()) == "EndcapTimingLayer") {
      isBarrel = false;
      edm::LogInfo("DD4hep_TestBTLPixelTopology") << "isBarrel = " << isBarrel;
    }

    theBaseNumber(fv);

    auto print_path = [&]() {
      std::stringstream ss;
      ss << " - OCMS[0]/";
      for (int ii = thisN_.getLevels() - 1; ii-- > 0;) {
        ss << thisN_.getLevelName(ii);
        ss << "[";
        ss << thisN_.getCopyNumber(ii);
        ss << "]/";
      }
      return ss.str();
    };

    if (level > 0 && fv.navPos().size() < level) {
      level = 0;
      exitLoop = true;
    }
    if (dd4hep::dd::noNamespace(fv.name()) == "BarrelTimingLayer") {
      level = fv.navPos().size();
    }

    // Test only the desired subdetector

    if (exitLoop && isBarrel) {
      break;
    }

    // Actions for MTD volumes: search for sensitive detectors

    bool isSens = false;

    for (auto const& t : specs) {
      for (auto const& it : t.second->paths) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(fv.name()), dd4hep::dd::realTopName(it))) {
          isSens = true;
          break;
        }
      }
    }

    if (isSens && isBarrel) {
      std::stringstream spix;
      spix << print_path() << "\n\n";

      BTLDetId theId(btlNS_.getUnitID(thisN_));

      DetId geoId = theId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()));
      const MTDGeomDet* thedet = pDG.product()->idToDet(geoId);

      if (thedet == nullptr) {
        throw cms::Exception("BTLDeviceSim") << "GeographicalID: " << std::hex << geoId.rawId() << " (" << theId.rawId()
                                             << ") is invalid!" << std::dec << std::endl;
      }
      const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

      int origRow = theId.row(topo.nrows());
      int origCol = theId.column(topo.nrows());
      spix << "rawId= " << theId.rawId() << " geoId= " << geoId.rawId() << " side/rod= " << theId.mtdSide() << " / "
           << theId.mtdRR() << " type/RU= " << theId.modType() << " / " << theId.runit()
           << " module/geomodule= " << theId.module() << " / " << static_cast<BTLDetId>(geoId).module()
           << " crys= " << theId.crystal() << " BTLDetId row/col= " << origRow << " / " << origCol;
      spix << "\n";

      //
      // Test of positions for sensitive detectors
      //

      auto fround = [&](double in) {
        std::stringstream ss;
        ss << std::fixed << std::setw(14) << roundIfNear0(in);
        return ss.str();
      };

      if (!dd4hep::isA<dd4hep::Box>(fv.solid())) {
        throw cms::Exception("DD4hep_TestBTLPixelTopology") << "MTD sensitive element not a DDBox";
        break;
      }
      dd4hep::Box mySens(fv.solid());

      DD3Vector zeroLocal(0., 0., 0.);
      DD3Vector cn1Local(mySens.x(), mySens.y(), mySens.z());
      DD3Vector cn2Local(-mySens.x(), -mySens.y(), -mySens.z());
      DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
      DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();
      DD3Vector cn2Global = (fv.rotation())(cn2Local) + fv.translation();

      const size_t nTest(3);
      std::array<Local3DPoint, nTest> refLocalPoints{{Local3DPoint(zeroLocal.x(), zeroLocal.y(), zeroLocal.z()),
                                                      Local3DPoint(cn1Local.x(), cn1Local.y(), cn1Local.z()),
                                                      Local3DPoint(cn2Local.x(), cn2Local.y(), cn2Local.z())}};
      std::array<DD3Vector, nTest> refGlobalPoints{{zeroGlobal, cn1Global, cn2Global}};

      for (size_t iloop = 0; iloop < nTest; iloop++) {
        // translate from crystal-local coordinates to module-local coordinates to get the row and column

        Local3DPoint cmRefLocal(convertMmToCm(refLocalPoints[iloop].x() / dd4hep::mm),
                                convertMmToCm(refLocalPoints[iloop].y() / dd4hep::mm),
                                convertMmToCm(refLocalPoints[iloop].z() / dd4hep::mm));
        Local3DPoint modLocal = topo.pixelToModuleLocalPoint(cmRefLocal, origRow, origCol);
        const auto& thepixel = topo.pixel(modLocal);
        uint8_t recoRow(thepixel.first), recoCol(thepixel.second);

        if (origRow != recoRow || origCol != recoCol) {
          std::stringstream warnmsg;
          warnmsg << "DIFFERENCE row/col, orig= " << origRow << " " << origCol
                  << " reco= " << static_cast<uint32_t>(recoRow) << " " << static_cast<uint32_t>(recoCol) << "\n";
          spix << warnmsg.str();
          sunitt << warnmsg.str();
          recoRow = origRow;
          recoCol = origCol;
        }

        Local3DPoint recoRefLocal = topo.moduleToPixelLocalPoint(modLocal);

        // reconstructed global position from reco geometry and rectangluar MTD topology

        const auto& modGlobal = thedet->toGlobal(modLocal);

        const double deltax = convertCmToMm(modGlobal.x()) - (refGlobalPoints[iloop].x() / dd4hep::mm);
        const double deltay = convertCmToMm(modGlobal.y()) - (refGlobalPoints[iloop].y() / dd4hep::mm);
        const double deltaz = convertCmToMm(modGlobal.z()) - (refGlobalPoints[iloop].z() / dd4hep::mm);

        const double local_deltax = recoRefLocal.x() - cmRefLocal.x();
        const double local_deltay = recoRefLocal.y() - cmRefLocal.y();
        const double local_deltaz = recoRefLocal.z() - cmRefLocal.z();

        spix << "Ref#" << iloop << " local= " << fround(refLocalPoints[iloop].x() / dd4hep::mm)
             << fround(refLocalPoints[iloop].y() / dd4hep::mm) << fround(refLocalPoints[iloop].z() / dd4hep::mm)
             << " Orig global= " << fround(refGlobalPoints[iloop].x() / dd4hep::mm)
             << fround(refGlobalPoints[iloop].y() / dd4hep::mm) << fround(refGlobalPoints[iloop].z() / dd4hep::mm)
             << " Reco global= " << fround(convertCmToMm(modGlobal.x())) << fround(convertCmToMm(modGlobal.y()))
             << fround(convertCmToMm(modGlobal.z())) << " Delta= " << fround(deltax) << fround(deltay) << fround(deltaz)
             << " Local Delta= " << fround(local_deltax) << fround(local_deltay) << fround(local_deltaz) << "\n";
        if (std::abs(deltax) > tolerance || std::abs(deltay) > tolerance || std::abs(deltaz) > tolerance) {
          std::stringstream warnmsg;
          warnmsg << "DIFFERENCE detId/ref# " << theId.rawId() << " " << iloop << " dx/dy/dz= " << fround(deltax)
                  << fround(deltay) << fround(deltaz) << "\n";
          spix << warnmsg.str();
          sunitt << warnmsg.str();
        }
        if (std::abs(local_deltax) > tolerance || std::abs(local_deltay) > tolerance ||
            std::abs(local_deltaz) > tolerance) {
          std::stringstream warnmsg;
          warnmsg << "DIFFERENCE detId/ref# " << theId.rawId() << " " << iloop
                  << " local dx/dy/dz= " << fround(local_deltax) << fround(local_deltay) << fround(local_deltaz)
                  << "\n";
          spix << warnmsg.str();
          sunitt << warnmsg.str();
        }
      }

      spix << "\n";
      edm::LogVerbatim("DD4hep_TestBTLPixelTopology") << spix.str();
    }
  } while (fv.next(0));

  if (!sunitt.str().empty()) {
    edm::LogVerbatim("MTDUnitTest") << sunitt.str();
  }
}

void DD4hep_TestBTLPixelTopology::theBaseNumber(cms::DDFilteredView& fv) {
  thisN_.reset();
  thisN_.setSize(fv.navPos().size());

  for (uint ii = 0; ii < fv.navPos().size(); ii++) {
    std::string_view name((fv.geoHistory()[ii])->GetName());
    size_t ipos = name.rfind('_');
    thisN_.addLevel(name.substr(0, ipos), fv.copyNos()[ii]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_TestBTLPixelTopology") << name.substr(0, ipos) << " " << fv.copyNos()[ii];
#endif
  }
}

DEFINE_FWK_MODULE(DD4hep_TestBTLPixelTopology);
