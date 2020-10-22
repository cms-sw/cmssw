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

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Rounding.h"

//#define EDM_ML_DEBUG

using namespace cms;

class DD4hep_TestMTDIdealGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestMTDIdealGeometry(const edm::ParameterSet&);
  ~DD4hep_TestMTDIdealGeometry() = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(cms::DDFilteredView& fv);

private:
  const edm::ESInputTag tag_;
  std::string ddTopNodeName_;
  uint32_t theLayout_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
};

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using angle_units::operators::convertRadToDeg;
using cms_rounding::roundIfNear0;
using geant_units::operators::convertCmToMm;

DD4hep_TestMTDIdealGeometry::DD4hep_TestMTDIdealGeometry(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      theLayout_(iConfig.getUntrackedParameter<uint32_t>("theLayout", 1)),
      thisN_(),
      btlNS_(),
      etlNS_() {
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
}

void DD4hep_TestMTDIdealGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

  if (ddTopNodeName_ != "BarrelTimingLayer" && ddTopNodeName_ != "EndcapTimingLayer") {
    edm::LogWarning("DD4hep_TestMTDIdealGeometry") << ddTopNodeName_ << "Not valid top MTD volume";
    return;
  }

  if (!pDD.isValid()) {
    edm::LogError("DD4hep_TestMTDIdealGeometry") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("DD4hep_TestMTDIdealGeometry") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("DD4hep_TestMTDIdealGeometry") << "NO label found pDD.description() returned false.";
  }

  if (!pSP.isValid()) {
    edm::LogError("DD4hep_TestMTDIdealGeometry") << "ESTransientHandle<DDSpecParRegistry> pSP is not valid!";
    return;
  }

  DDFilteredView fv(pDD.product(), pDD.product()->description()->worldVolume());
  fv.next(0);
  edm::LogInfo("DD4hep_TestMTDIdealGeometry") << fv.name();

  DDSpecParRefs specs;
  std::string attribute("ReadOutName"), name;
  if (ddTopNodeName_ == "BarrelTimingLayer") {
    name = "FastTimerHitsBarrel";
  } else if (ddTopNodeName_ == "EndcapTimingLayer") {
    name = "FastTimerHitsEndcap";
  }
  if (name.empty()) {
    edm::LogError("DD4hep_TestMTDIdealGeometry") << "No sensitive detector provided, abort";
    return;
  }
  pSP.product()->filter(specs, attribute, name);

  edm::LogVerbatim("Geometry").log([&specs](auto& log) {
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

  bool write = false;
  bool isBarrel = true;
  bool exitLoop = false;
  uint32_t level(0);

  do {
    if (fv.name() == "BarrelTimingLayer") {
      isBarrel = true;
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
    } else if (fv.name() == "EndcapTimingLayer") {
      isBarrel = false;
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
    }

    std::stringstream ss;

    theBaseNumber(fv);

    auto print_path = [&]() {
      ss << " - OCMS[0]/";
      for (int ii = thisN_.getLevels() - 1; ii-- > 0;) {
        ss << thisN_.getLevelName(ii);
        ss << "[";
        ss << thisN_.getCopyNumber(ii);
        ss << "]/";
      }
    };

    if (level > 0 && fv.navPos().size() < level) {
      level = 0;
      write = false;
      exitLoop = true;
    }
    if (fv.name() == ddTopNodeName_) {
      write = true;
      level = fv.navPos().size();
    }

    // Test only the desired subdetector

    if (exitLoop && isBarrel) {
      break;
    }

    // Actions for MTD volumes: searchg for sensitive detectors

    if (write) {
      print_path();

#ifdef EDM_ML_DEBUG
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << fv.path();
#endif

      edm::LogInfo("DD4hep_TestMTDPath") << ss.str();

      bool isSens = false;

      for (auto const& t : specs) {
        for (auto const& it : t.second->paths) {
          if (dd4hep::dd::compareEqual(fv.name(), dd4hep::dd::realTopName(it))) {
            isSens = true;
            break;
          }
        }
      }

      if (isSens) {
        //
        // Test of numbering scheme for sensitive detectors
        //

        std::stringstream sunitt;
        std::stringstream snum;

        if (isBarrel) {
          BTLDetId::CrysLayout lay = static_cast<BTLDetId::CrysLayout>(theLayout_);
          BTLDetId theId(btlNS_.getUnitID(thisN_));
          int hIndex = theId.hashedIndex(lay);
          BTLDetId theNewId(theId.getUnhashedIndex(hIndex, lay));
          sunitt << theId.rawId();
          snum << theId << "\n layout type = " << static_cast<int>(lay) << "\n ieta        = " << theId.ieta(lay)
               << "\n iphi        = " << theId.iphi(lay) << "\n hashedIndex = " << theId.hashedIndex(lay)
               << "\n BTLDetId hI = " << theNewId;
          if (theId.mtdSide() != theNewId.mtdSide()) {
            snum << "\n DIFFERENCE IN SIDE";
          }
          if (theId.mtdRR() != theNewId.mtdRR()) {
            snum << "\n DIFFERENCE IN ROD";
          }
          if (theId.module() != theNewId.module()) {
            snum << "\n DIFFERENCE IN MODULE";
          }
          if (theId.modType() != theNewId.modType()) {
            snum << "\n DIFFERENCE IN MODTYPE";
          }
          if (theId.crystal() != theNewId.crystal()) {
            snum << "\n DIFFERENCE IN CRYSTAL";
          }
          snum << "\n";
        } else {
          ETLDetId theId(etlNS_.getUnitID(thisN_));
          sunitt << theId.rawId();
          snum << theId;
        }
        edm::LogInfo("DD4hep_TestMTDNumbering") << snum.str();

        //
        // Test of positions for sensitive detectors
        //

        std::stringstream spos;

        auto fround = [&](double in) {
          std::stringstream ss;
          ss << std::fixed << std::setw(14) << roundIfNear0(in);
          return ss.str();
        };

        if (!dd4hep::isA<dd4hep::Box>(fv.solid())) {
          throw cms::Exception("TestMTDIdealGeometry") << "MTD sensitive element not a DDBox";
          break;
        }
        dd4hep::Box mySens(fv.solid());
        spos << "Solid shape name: " << DDSolidShapesName::name(fv.legacyShape(fv.shape())) << "\n";
        spos << "Box dimensions: " << fround(convertCmToMm(mySens.x())) << " " << fround(convertCmToMm(mySens.y()))
             << " " << fround(convertCmToMm(mySens.z())) << "\n";

        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        spos << "Translation vector components: " << fround(convertCmToMm(fv.translation().x())) << " "
             << fround(convertCmToMm(fv.translation().y())) << " " << fround(convertCmToMm(fv.translation().z())) << " "
             << "\n";
        spos << "Rotation matrix components: " << fround(x.X()) << " " << fround(x.Y()) << " " << fround(x.Z()) << " "
             << fround(y.X()) << " " << fround(y.Y()) << " " << fround(y.Z()) << " " << fround(z.X()) << " "
             << fround(z.Y()) << " " << fround(z.Z()) << "\n";

        DD3Vector zeroLocal(0., 0., 0.);
        DD3Vector cn1Local(mySens.x(), mySens.y(), mySens.z());
        double distLocal = cn1Local.R();
        DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
        DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();
        double distGlobal =
            std::sqrt(std::pow(zeroGlobal.X() - cn1Global.X(), 2) + std::pow(zeroGlobal.Y() - cn1Global.Y(), 2) +
                      std::pow(zeroGlobal.Z() - cn1Global.Z(), 2));

        spos << "Center global   = " << fround(convertCmToMm(zeroGlobal.X())) << fround(convertCmToMm(zeroGlobal.Y()))
             << fround(convertCmToMm(zeroGlobal.Z())) << " r = " << fround(convertCmToMm(zeroGlobal.Rho()))
             << " phi = " << fround(convertRadToDeg(zeroGlobal.Phi())) << "\n";

        spos << "Corner 1 local  = " << fround(convertCmToMm(cn1Local.X())) << fround(convertCmToMm(cn1Local.Y()))
             << fround(convertCmToMm(cn1Local.Z())) << " DeltaR = " << fround(convertCmToMm(distLocal)) << "\n";

        spos << "Corner 1 global = " << fround(convertCmToMm(cn1Global.X())) << fround(convertCmToMm(cn1Global.Y()))
             << fround(convertCmToMm(cn1Global.Z())) << " DeltaR = " << fround(convertCmToMm(distGlobal)) << "\n";

        spos << "\n";
        if (std::fabs(convertCmToMm(distGlobal - distLocal)) > 1.e-6) {
          spos << "DIFFERENCE IN DISTANCE \n";
        }
        sunitt << fround(convertCmToMm(zeroGlobal.X())) << fround(convertCmToMm(zeroGlobal.Y()))
               << fround(convertCmToMm(zeroGlobal.Z())) << fround(convertCmToMm(cn1Global.X()))
               << fround(convertCmToMm(cn1Global.Y())) << fround(convertCmToMm(cn1Global.Z()));
        edm::LogInfo("DD4hep_TestMTDPosition") << spos.str();

        edm::LogVerbatim("MTDUnitTest") << sunitt.str();
      }
    }
  } while (fv.next(0));
}

void DD4hep_TestMTDIdealGeometry::theBaseNumber(cms::DDFilteredView& fv) {
  thisN_.reset();
  thisN_.setSize(fv.navPos().size());

  for (uint ii = 0; ii < fv.navPos().size(); ii++) {
    std::string name((fv.geoHistory()[ii])->GetName());
    name.assign(name.erase(name.rfind('_')));
    int copyN(fv.copyNos()[ii]);
    thisN_.addLevel(name, copyN);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_TestMTDIdealGeometry") << name << " " << copyN;
#endif
  }
}

DEFINE_FWK_MODULE(DD4hep_TestMTDIdealGeometry);
