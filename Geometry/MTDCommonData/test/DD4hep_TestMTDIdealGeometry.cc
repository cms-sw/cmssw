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

#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/Math/interface/Rounding.h"
#include <DD4hep/DD4hepUnits.h>

//#define EDM_ML_DEBUG

using namespace cms;

class DD4hep_TestMTDIdealGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestMTDIdealGeometry(const edm::ParameterSet&);
  ~DD4hep_TestMTDIdealGeometry() override = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(cms::DDFilteredView& fv);

private:
  const edm::ESInputTag tag_;
  std::string ddTopNodeName_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
};

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using angle_units::operators::convertRadToDeg;
using cms_rounding::roundIfNear0;

DD4hep_TestMTDIdealGeometry::DD4hep_TestMTDIdealGeometry(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
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
    if (dd4hep::dd::noNamespace(fv.name()) == "BarrelTimingLayer") {
      isBarrel = true;
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
    } else if (dd4hep::dd::noNamespace(fv.name()) == "EndcapTimingLayer") {
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
    if (dd4hep::dd::noNamespace(fv.name()) == ddTopNodeName_) {
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
          if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(fv.name()), dd4hep::dd::realTopName(it))) {
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
          BTLDetId theId(btlNS_.getUnitID(thisN_));
          sunitt << theId.rawId();
          snum << theId;
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
        spos << "Box dimensions: " << fround(mySens.x() / dd4hep::mm) << " " << fround(mySens.y() / dd4hep::mm) << " "
             << fround(mySens.z() / dd4hep::mm) << "\n";

        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        spos << "Translation vector components: " << fround(fv.translation().x() / dd4hep::mm) << " "
             << fround(fv.translation().y() / dd4hep::mm) << " " << fround(fv.translation().z() / dd4hep::mm) << " "
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

        spos << "Center global   = " << fround(zeroGlobal.X() / dd4hep::mm) << fround(zeroGlobal.Y() / dd4hep::mm)
             << fround(zeroGlobal.Z() / dd4hep::mm) << " r = " << fround(zeroGlobal.Rho() / dd4hep::mm)
             << " phi = " << fround(convertRadToDeg(zeroGlobal.Phi())) << "\n";

        spos << "Corner 1 local  = " << fround(cn1Local.X() / dd4hep::mm) << fround(cn1Local.Y() / dd4hep::mm)
             << fround(cn1Local.Z() / dd4hep::mm) << " DeltaR = " << fround(distLocal / dd4hep::mm) << "\n";

        spos << "Corner 1 global = " << fround(cn1Global.X() / dd4hep::mm) << fround(cn1Global.Y() / dd4hep::mm)
             << fround(cn1Global.Z() / dd4hep::mm) << " DeltaR = " << fround(distGlobal / dd4hep::mm) << "\n";

        spos << "\n";
        if (std::fabs(distGlobal - distLocal) / dd4hep::mm > 1.e-6) {
          spos << "DIFFERENCE IN DISTANCE \n";
        }
        sunitt << fround(zeroGlobal.X() / dd4hep::mm) << fround(zeroGlobal.Y() / dd4hep::mm)
               << fround(zeroGlobal.Z() / dd4hep::mm) << fround(cn1Global.X() / dd4hep::mm)
               << fround(cn1Global.Y() / dd4hep::mm) << fround(cn1Global.Z() / dd4hep::mm);
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
    std::string_view name((fv.geoHistory()[ii])->GetName());
    size_t ipos = name.rfind('_');
    thisN_.addLevel(name.substr(0, ipos), fv.copyNos()[ii]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_TestMTDIdealGeometry") << name.substr(0, ipos) << " " << fv.copyNos()[ii];
#endif
  }
}

DEFINE_FWK_MODULE(DD4hep_TestMTDIdealGeometry);
