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
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Rounding.h"
#include "DataFormats/Math/interface/angle_units.h"
#include <DD4hep/DD4hepUnits.h>

using namespace cms;
using namespace geant_units::operators;
using namespace cms_rounding;

class DD4hep_TestPixelTopology : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestPixelTopology(const edm::ParameterSet&);
  ~DD4hep_TestPixelTopology() = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(cms::DDFilteredView& fv);

private:
  inline std::string fround(const double in, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  }

  inline std::string fvecround(const auto& vecin, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  }

  void analyseRectangle(const GeomDetUnit& det);
  void checkRotation(const GeomDetUnit& det);

  const edm::ESInputTag tag_;
  std::string ddTopNodeName_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;

  std::stringstream sunitt_;
  constexpr static double tolerance{0.5e-3_mm};
};

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using angle_units::operators::convertRadToDeg;

DD4hep_TestPixelTopology::DD4hep_TestPixelTopology(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      thisN_(),
      btlNS_(),
      etlNS_() {
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>(tag_);
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>(tag_);
}

void DD4hep_TestPixelTopology::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

  if (ddTopNodeName_ != "BarrelTimingLayer" && ddTopNodeName_ != "EndcapTimingLayer") {
    edm::LogWarning("DD4hep_TestPixelTopology") << ddTopNodeName_ << "Not valid top MTD volume";
    return;
  }

  if (!pDD.isValid()) {
    edm::LogError("DD4hep_TestPixelTopology") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogVerbatim("DD4hep_TestPixelTopology") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogPrint("DD4hep_TestPixelTopology") << "NO label found pDD.description() returned false.";
  }

  if (!pSP.isValid()) {
    edm::LogError("DD4hep_TestPixelTopology") << "ESTransientHandle<DDSpecParRegistry> pSP is not valid!";
    return;
  }

  auto pTP = iSetup.getTransientHandle(mtdtopoToken_);
  if (!pTP.isValid()) {
    edm::LogError("DD4hep_TestPixelTopology") << "ESTransientHandle<MTDTopology> pTP is not valid!";
    return;
  } else {
    edm::LogVerbatim("DD4hep_TestPixelTopology")
        << "MTD topology mode = " << pTP.product()->getMTDTopologyMode() << " BtlLayout = "
        << static_cast<int>(MTDTopologyMode::crysLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()))
        << " EtlLayout = "
        << static_cast<int>(MTDTopologyMode::etlLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()));
    sunitt_ << "MTD topology mode = " << pTP.product()->getMTDTopologyMode() << " BtlLayout = "
            << static_cast<int>(MTDTopologyMode::crysLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()))
            << " EtlLayout = "
            << static_cast<int>(MTDTopologyMode::etlLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()));
  }

  auto pDG = iSetup.getTransientHandle(mtdgeoToken_);
  if (!pDG.isValid()) {
    edm::LogError("DD4hep_TestPixelTopology") << "ESTransientHandle<MTDGeometry> pDG is not valid!";
    return;
  }

  DDSpecParRefs specs;
  std::string attribute("MtdDDStructure"), name;
  bool isBarrel = false;
  if (ddTopNodeName_ == "BarrelTimingLayer") {
    edm::LogVerbatim("DD4hep_TestPixelTopology") << "  BTL MTDGeometry:\n";
    sunitt_ << "  BTL MTDGeometry:\n";
    name = "FastTimerHitsBarrel";
    isBarrel = true;
  } else if (ddTopNodeName_ == "EndcapTimingLayer") {
    edm::LogVerbatim("DD4hep_TestPixelTopology") << "  ETL MTDGeometry:\n";
    sunitt_ << "  ETL MTDGeometry:\n";
    name = "FastTimerHitsEndcap";
  } else {
    edm::LogError("DD4hep_TestPixelTopology") << "No correct sensitive detector provided, abort" << ddTopNodeName_;
    return;
  }
  pSP.product()->filter(specs, attribute, ddTopNodeName_);
  attribute = "ReadOutName";
  pSP.product()->filter(specs, attribute, name);

  edm::LogVerbatim("DD4hep_TestPixelTopology").log([&specs](auto& log) {
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

  std::vector<std::string_view> filterName;
  for (auto const& t : specs) {
    for (auto const& kl : t.second->spars) {
      if (kl.first == attribute) {
        for (auto const& it : t.second->paths) {
          filterName.emplace_back(it);
        }
      }
    }
  }

  DDFilteredView fv(pDD.product(), pDD.product()->description()->worldVolume());
  fv.mergedSpecifics(specs);
  fv.firstChild();

  bool write = false;
  bool exitLoop = false;
  uint32_t level(0);
  uint32_t count(0);
  uint32_t nSensBTL(0);
  uint32_t nSensETL(0);
  uint32_t oldgeoId(0);

  do {
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

    if (level > 0 && fv.navPos().size() < level && count == 2) {
      exitLoop = true;
    }
    if (dd4hep::dd::noNamespace(fv.name()) == ddTopNodeName_) {
      write = true;
      level = fv.navPos().size();
      count++;
    }

    // Test only the desired subdetector

    if (exitLoop) {
      break;
    }

    if (write) {
      // Actions for MTD volumes: search for sensitive detectors

      bool isSens = false;

      for (auto const& it : filterName) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(fv.name()), dd4hep::dd::realTopName(it))) {
          isSens = true;
          break;
        }
      }

      if (isSens) {
        DetId theId, geoId;
        BTLDetId theIdBTL, modIdBTL;
        ETLDetId theIdETL, modIdETL;
        if (isBarrel) {
          theIdBTL = btlNS_.getUnitID(thisN_);
          theId = theIdBTL;
          geoId = theIdBTL.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(pTP.product()->getMTDTopologyMode()));
          modIdBTL = geoId;
        } else {
          theIdETL = etlNS_.getUnitID(thisN_);
          theId = theIdETL;
          geoId = theIdETL.geographicalId();
          modIdETL = geoId;
        }

        const MTDGeomDet* thedet = pDG.product()->idToDet(geoId);

        if (dynamic_cast<const MTDGeomDetUnit*>((thedet)) == nullptr) {
          throw cms::Exception("DD4hep_TestPixelTopology")
              << "GeographicalID: " << std::hex << geoId.rawId() << " (" << theId.rawId()
              << ") with invalid MTDGeomDetUnit!" << std::dec << std::endl;
        }

        bool isNewId(false);
        if (geoId != oldgeoId) {
          oldgeoId = geoId;
          isNewId = true;
          if (isBarrel) {
            nSensBTL++;
          } else {
            nSensETL++;
          }
          const GeomDetUnit theDetUnit = *(dynamic_cast<const MTDGeomDetUnit*>(thedet));

          if (isBarrel) {
            edm::LogVerbatim("DD4hep_TestPixelTopology")
                << "geoId= " << modIdBTL.rawId() << " side= " << modIdBTL.mtdSide()
                << " RU/mod= " << modIdBTL.globalRunit() << " / " << modIdBTL.module();
            sunitt_ << "geoId= " << modIdBTL.rawId() << " side= " << modIdBTL.mtdSide()
                    << " RU/mod= " << modIdBTL.globalRunit() << " / " << modIdBTL.module();
          } else {
            edm::LogVerbatim("DD4hep_TestPixelTopology")
                << "geoId= " << modIdETL.rawId() << " side= " << modIdETL.mtdSide()
                << " disc/face/sec= " << modIdETL.nDisc() << " / " << modIdETL.discSide() << " / " << modIdETL.sector()
                << " mod/typ/sens= " << modIdETL.module() << " / " << modIdETL.modType() << " / " << modIdETL.sensor();
            sunitt_ << "geoId= " << modIdETL.rawId() << " side= " << modIdETL.mtdSide()
                    << " disc/face/sec= " << modIdETL.nDisc() << " / " << modIdETL.discSide() << " / "
                    << modIdETL.sector() << " mod/typ/sens= " << modIdETL.module() << " / " << modIdETL.modType()
                    << " / " << modIdETL.sensor();
          }
          analyseRectangle(theDetUnit);
        }

        if (thedet == nullptr) {
          throw cms::Exception("DD4hep_TestPixelTopology") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                           << theId.rawId() << ") is invalid!" << std::dec << std::endl;
        }
        const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
        const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

        int origRow(-1), origCol(-1), recoRow(-1), recoCol(-1);
        if (isBarrel) {
          origRow = theIdBTL.row(topo.nrows());
          origCol = theIdBTL.column(topo.nrows());
        }

        //
        // Test of positions for sensitive detectors
        //

        if (!dd4hep::isA<dd4hep::Box>(fv.solid())) {
          throw cms::Exception("DD4hep_TestPixelTopology") << "MTD sensitive element not a DDBox";
          break;
        }
        dd4hep::Box mySens(fv.solid());

        double xoffset(0.);
        double yoffset(0.);
        if (!isBarrel) {
          xoffset = topo.gapxBorder() + 0.5 * topo.pitch().first;
          yoffset = topo.gapyBorder() + 0.5 * topo.pitch().second;
        }
        DD3Vector zeroLocal(0., 0., 0.);
        DD3Vector cn1Local(mySens.x() - xoffset, mySens.y() - yoffset, mySens.z());
        DD3Vector cn2Local(-mySens.x() + xoffset, -mySens.y() + yoffset, -mySens.z());
        DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
        DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();
        DD3Vector cn2Global = (fv.rotation())(cn2Local) + fv.translation();

        const size_t nTest(3);
        std::array<Local3DPoint, nTest> refLocalPoints{{Local3DPoint(zeroLocal.x(), zeroLocal.y(), zeroLocal.z()),
                                                        Local3DPoint(cn1Local.x(), cn1Local.y(), cn1Local.z()),
                                                        Local3DPoint(cn2Local.x(), cn2Local.y(), cn2Local.z())}};
        std::array<DD3Vector, nTest> refGlobalPoints{{zeroGlobal, cn1Global, cn2Global}};

        for (size_t iloop = 0; iloop < nTest; iloop++) {
          Local3DPoint cmRefLocal(convertMmToCm(refLocalPoints[iloop].x() / dd4hep::mm),
                                  convertMmToCm(refLocalPoints[iloop].y() / dd4hep::mm),
                                  convertMmToCm(refLocalPoints[iloop].z() / dd4hep::mm));

          Local3DPoint modLocal, recoRefLocal;
          if (isBarrel) {
            // if BTL translate from crystal-local coordinates to module-local coordinates to get the row and column
            modLocal = topo.pixelToModuleLocalPoint(cmRefLocal, origRow, origCol);
            recoRefLocal = topo.moduleToPixelLocalPoint(modLocal);
            const auto& thepixel = topo.pixelIndex(modLocal);
            recoRow = thepixel.first;
            recoCol = thepixel.second;

            if (origRow != recoRow || origCol != recoCol) {
              edm::LogVerbatim("DD4hep_TestPixelTopology") << "DIFFERENCE row/col, orig= " << origRow << " " << origCol
                                                           << " reco= " << recoRow << " " << recoCol << "\n";
              sunitt_ << "DIFFERENCE row/col, orig= " << origRow << " " << origCol << " reco= " << recoRow << " "
                      << recoCol << "\n";
              recoRow = origRow;
              recoCol = origCol;
            }
          } else {
            // if ETL find the pixel corresponding to the referemce point, compute the pixel coordinate and convert back for check
            modLocal = cmRefLocal;
            const auto& thepixel = topo.pixelIndex(modLocal);
            Local3DPoint pixLocal = topo.moduleToPixelLocalPoint(modLocal);
            recoRefLocal = topo.pixelToModuleLocalPoint(pixLocal, thepixel.first, thepixel.second);
            recoRow = thepixel.first;
            recoCol = thepixel.second;
          }

          // reconstructed global position from reco geometry and rectangluar MTD topology

          const auto& modGlobal = thedet->toGlobal(modLocal);

          if (isNewId && iloop == nTest - 1) {
            edm::LogVerbatim("DD4hep_TestPixelTopology")
                << "row/col= " << recoRow << " / " << recoCol << " local pos= " << fvecround(modLocal, 4)
                << " global pos= " << fvecround(modGlobal, 4) << "\n";
            sunitt_ << "row/col= " << recoRow << " / " << recoCol << " local pos= " << fvecround(modLocal, 2)
                    << " global pos= " << fvecround(modGlobal, 2) << "\n";
          }

          const double deltax = convertCmToMm(modGlobal.x()) - (refGlobalPoints[iloop].x() / dd4hep::mm);
          const double deltay = convertCmToMm(modGlobal.y()) - (refGlobalPoints[iloop].y() / dd4hep::mm);
          const double deltaz = convertCmToMm(modGlobal.z()) - (refGlobalPoints[iloop].z() / dd4hep::mm);

          const double local_deltax = recoRefLocal.x() - cmRefLocal.x();
          const double local_deltay = recoRefLocal.y() - cmRefLocal.y();
          const double local_deltaz = recoRefLocal.z() - cmRefLocal.z();

          if (std::abs(deltax) > tolerance || std::abs(deltay) > tolerance || std::abs(deltaz) > tolerance ||
              std::abs(local_deltax) > tolerance || std::abs(local_deltay) > tolerance ||
              std::abs(local_deltaz) > tolerance) {
            edm::LogVerbatim("DD4hep_TestPixelTopology") << print_path() << "\n";
            sunitt_ << print_path() << "\n";
            if (isBarrel) {
              edm::LogVerbatim("DD4hep_TestPixelTopology")
                  << "rawId= " << theIdBTL.rawId() << " geoId= " << geoId.rawId() << " side/rod= " << theIdBTL.mtdSide()
                  << " / " << theIdBTL.mtdRR() << " RU= " << theIdBTL.globalRunit()
                  << " module/geomodule= " << theIdBTL.module() << " / " << static_cast<BTLDetId>(geoId).module()
                  << " crys= " << theIdBTL.crystal() << " BTLDetId row/col= " << origRow << " / " << origCol << "\n";
              sunitt_ << "rawId= " << theIdBTL.rawId() << " geoId= " << geoId.rawId()
                      << " side/rod= " << theIdBTL.mtdSide() << " / " << theIdBTL.mtdRR()
                      << " RU= " << theIdBTL.globalRunit() << " module/geomodule= " << theIdBTL.module() << " / "
                      << static_cast<BTLDetId>(geoId).module() << " crys= " << theIdBTL.crystal()
                      << " BTLDetId row/col= " << origRow << " / " << origCol << "\n";
            } else {
              edm::LogVerbatim("DD4hep_TestPixelTopology")
                  << "geoId= " << modIdETL.rawId() << " side= " << modIdETL.mtdSide()
                  << " disc/face/sec= " << modIdETL.nDisc() << " / " << modIdETL.discSide() << " / "
                  << modIdETL.sector() << " mod/typ/sens= " << modIdETL.module() << " / " << modIdETL.modType() << " / "
                  << modIdETL.sensor() << "\n";
              sunitt_ << "geoId= " << modIdETL.rawId() << " side= " << modIdETL.mtdSide()
                      << " disc/face/sec= " << modIdETL.nDisc() << " / " << modIdETL.discSide() << " / "
                      << modIdETL.sector() << " mod/typ/sens= " << modIdETL.module() << " / " << modIdETL.modType()
                      << " / " << modIdETL.sensor() << "\n";
            }

            edm::LogVerbatim("DD4hep_TestPixelTopology")
                << "Ref#" << iloop << " local= " << fround(refLocalPoints[iloop].x() / dd4hep::mm, 4)
                << fround(refLocalPoints[iloop].y() / dd4hep::mm, 4)
                << fround(refLocalPoints[iloop].z() / dd4hep::mm, 4)
                << " Orig global= " << fround(refGlobalPoints[iloop].x() / dd4hep::mm, 4)
                << fround(refGlobalPoints[iloop].y() / dd4hep::mm, 4)
                << fround(refGlobalPoints[iloop].z() / dd4hep::mm, 4)
                << " Reco global= " << fround(convertCmToMm(modGlobal.x()), 4)
                << fround(convertCmToMm(modGlobal.y()), 4) << fround(convertCmToMm(modGlobal.z()), 4)
                << " Delta= " << fround(deltax, 4) << fround(deltay, 4) << fround(deltaz, 4)
                << " Local Delta= " << fround(local_deltax, 4) << fround(local_deltay, 4) << fround(local_deltaz, 4)
                << "\n";
            sunitt_ << "Ref#" << iloop << " local= " << fround(refLocalPoints[iloop].x() / dd4hep::mm, 2)
                    << fround(refLocalPoints[iloop].y() / dd4hep::mm, 2)
                    << fround(refLocalPoints[iloop].z() / dd4hep::mm, 2)
                    << " Orig global= " << fround(refGlobalPoints[iloop].x() / dd4hep::mm, 2)
                    << fround(refGlobalPoints[iloop].y() / dd4hep::mm, 2)
                    << fround(refGlobalPoints[iloop].z() / dd4hep::mm, 2)
                    << " Reco global= " << fround(convertCmToMm(modGlobal.x()), 2)
                    << fround(convertCmToMm(modGlobal.y()), 2) << fround(convertCmToMm(modGlobal.z()), 2)
                    << " Delta= " << fround(deltax, 2) << fround(deltay, 2) << fround(deltaz, 2)
                    << " Local Delta= " << fround(local_deltax, 2) << fround(local_deltay, 2) << fround(local_deltaz, 2)
                    << "\n";

            if (std::abs(deltax) > tolerance || std::abs(deltay) > tolerance || std::abs(deltaz) > tolerance) {
              edm::LogVerbatim("DD4hep_TestPixelTopology")
                  << "DIFFERENCE detId/ref# " << theId.rawId() << " " << iloop << " dx/dy/dz= " << fround(deltax, 4)
                  << fround(deltay, 4) << fround(deltaz, 4) << "\n";
              sunitt_ << "DIFFERENCE detId/ref# " << theId.rawId() << " " << iloop << " dx/dy/dz= " << fround(deltax, 2)
                      << fround(deltay, 2) << fround(deltaz, 2) << "\n";
            }
            if (std::abs(local_deltax) > tolerance || std::abs(local_deltay) > tolerance ||
                std::abs(local_deltaz) > tolerance) {
              edm::LogVerbatim("DD4hep_TestPixelTopology")
                  << "DIFFERENCE detId/ref# " << theId.rawId() << " " << iloop
                  << " local dx/dy/dz= " << fround(local_deltax, 4) << fround(local_deltay, 4)
                  << fround(local_deltaz, 4) << "\n";
              sunitt_ << "DIFFERENCE detId/ref# " << theId.rawId() << " " << iloop
                      << " local dx/dy/dz= " << fround(local_deltax, 2) << fround(local_deltay, 2)
                      << fround(local_deltaz, 2) << "\n";
            }
          }
        }
      }
    }
  } while (fv.next(0));

  if (isBarrel && nSensBTL != pDG.product()->detsBTL().size()) {
    edm::LogVerbatim("DD4hep_TestPixelTopology")
        << "DIFFERENCE #ideal = " << nSensBTL << " #reco = " << pDG.product()->detsBTL().size()
        << " BTL module numbers are not matching!";
    sunitt_ << "DIFFERENCE #ideal = " << nSensBTL << " #reco = " << pDG.product()->detsBTL().size()
            << " BTL module numbers are not matching!";
  }

  if (!isBarrel && nSensETL != pDG.product()->detsETL().size()) {
    edm::LogVerbatim("DD4hep_TestPixelTopology")
        << "DIFFERENCE #ideal = " << nSensETL << " #reco = " << pDG.product()->detsBTL().size()
        << " ETL module numbers are not matching!";
    sunitt_ << "DIFFERENCE #ideal = " << nSensETL << " #reco = " << pDG.product()->detsBTL().size()
            << " ETL module numbers are not matching!";
  }

  if (!sunitt_.str().empty()) {
    edm::LogVerbatim("MTDUnitTest") << sunitt_.str();
  }
}

void DD4hep_TestPixelTopology::theBaseNumber(cms::DDFilteredView& fv) {
  thisN_.reset();
  thisN_.setSize(fv.navPos().size());

  for (uint ii = 0; ii < fv.navPos().size(); ii++) {
    std::string_view name((fv.geoHistory()[ii])->GetName());
    size_t ipos = name.rfind('_');
    thisN_.addLevel(name.substr(0, ipos), fv.copyNos()[ii]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_TestPixelTopology") << ii << " " << name.substr(0, ipos) << " " << fv.copyNos()[ii];
#endif
  }
}

void DD4hep_TestPixelTopology::analyseRectangle(const GeomDetUnit& det) {
  const double safety = 0.9999;

  const BoundPlane& p = det.specificSurface();
  const Bounds& bounds = det.surface().bounds();
  const RectangularPlaneBounds* tb = dynamic_cast<const RectangularPlaneBounds*>(&bounds);
  if (tb == nullptr)
    return;  // not trapezoidal

  const GlobalPoint& pos = det.position();
  double length = tb->length();
  double width = tb->width();
  double thickness = tb->thickness();

  GlobalVector yShift = det.surface().toGlobal(LocalVector(0, 0, safety * length / 2.));
  GlobalPoint outerMiddle = pos + yShift;
  GlobalPoint innerMiddle = pos + (-1. * yShift);
  if (outerMiddle.perp() < innerMiddle.perp())
    std::swap(outerMiddle, innerMiddle);

  edm::LogVerbatim("DD4hep_TestPixelTopology")
      << " " << fvecround(pos, 4) << " R= " << fround(std::sqrt(pos.x() * pos.x() + pos.y() * pos.y()), 4)
      << " phi= " << fround(convertRadToDeg(pos.phi()), 4) << " outerMiddle " << fvecround(outerMiddle, 4) << "\n"
      << " l/w/t " << fround(length, 4) << " / " << fround(width, 4) << " / " << fround(thickness, 4)
      << " RadLeng= " << p.mediumProperties().radLen() << " Xi= " << p.mediumProperties().xi()
      << " det center inside bounds? " << tb->inside(det.surface().toLocal(pos)) << "\n";
  sunitt_ << " " << fvecround(pos, 2) << " R= " << fround(std::sqrt(pos.x() * pos.x() + pos.y() * pos.y()), 2)
          << " phi= " << fround(convertRadToDeg(pos.phi()), 2) << " outerMiddle " << fvecround(outerMiddle, 2) << "\n"
          << " l/w/t " << fround(length, 2) << " / " << fround(width, 2) << " / " << fround(thickness, 2)
          << " RadLeng= " << p.mediumProperties().radLen() << " Xi= " << p.mediumProperties().xi()
          << " det center inside bounds? " << tb->inside(det.surface().toLocal(pos)) << "\n";

  checkRotation(det);
}

void DD4hep_TestPixelTopology::checkRotation(const GeomDetUnit& det) {
  const double eps = 10. * std::numeric_limits<float>::epsilon();
  static int first = 0;
  if (first == 0) {
    edm::LogVerbatim("DD4hep_TestPixelTopology") << "factor x numeric_limits<float>::epsilon() " << eps;
    first = 1;
  }

  const Surface::RotationType& rot(det.surface().rotation());
  GlobalVector a(rot.xx(), rot.xy(), rot.xz());
  GlobalVector b(rot.yx(), rot.yy(), rot.yz());
  GlobalVector c(rot.zx(), rot.zy(), rot.zz());
  GlobalVector cref = a.cross(b);
  GlobalVector aref = b.cross(c);
  GlobalVector bref = c.cross(a);
  if ((a - aref).mag() > eps || (b - bref).mag() > eps || (c - cref).mag() > eps) {
    edm::LogVerbatim("DD4hep_TestPixelTopology")
        << " DIFFERENCE Rotation not good by cross product: " << (a - aref).mag() << ", " << (b - bref).mag() << ", "
        << (c - cref).mag() << " for det at pos " << det.surface().position() << "\n";
    sunitt_ << " DIFFERENCE Rotation not good by cross product: " << (a - aref).mag() << ", " << (b - bref).mag()
            << ", " << (c - cref).mag() << " for det at pos " << det.surface().position() << "\n";
  }
  if (fabs(a.mag() - 1.) > eps || fabs(b.mag() - 1.) > eps || fabs(c.mag() - 1.) > eps) {
    edm::LogVerbatim("DD4hep_TestPixelTopology")
        << " DIFFERENCE Rotation not good by vector mag: " << (a).mag() << ", " << (b).mag() << ", " << (c).mag()
        << " for det at pos " << det.surface().position() << "\n";
    sunitt_ << " DIFFERENCE Rotation not good by vector mag: " << (a).mag() << ", " << (b).mag() << ", " << (c).mag()
            << " for det at pos " << det.surface().position() << "\n";
  }
}

DEFINE_FWK_MODULE(DD4hep_TestPixelTopology);
