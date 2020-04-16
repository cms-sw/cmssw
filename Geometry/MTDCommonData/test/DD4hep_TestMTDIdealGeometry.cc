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
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/Math/interface/GeantUnits.h"

//#define EDM_ML_DEBUG

using namespace cms;

class DD4hep_TestMTDIdealGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestMTDIdealGeometry(const edm::ParameterSet&);
  ~DD4hep_TestMTDIdealGeometry() = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(const std::vector<std::pair<std::string, uint32_t>>& gh);

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
      log << "\nRegExps { ";
      for (const auto& ki : t->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t->spars) {
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
  std::vector<std::pair<std::string, uint32_t>> geoHistory;

  do {
    uint32_t clevel = fv.navPos().size();
    uint32_t ccopy = (clevel > 1 ? fv.copyNum() : 0);
    geoHistory.resize(clevel);
    geoHistory[clevel - 1] = std::pair<std::string, uint32_t>(static_cast<std::string>(fv.name()), ccopy);

    if (fv.name() == "BarrelTimingLayer") {
      isBarrel = true;
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
    } else if (fv.name() == "EndcapTimingLayer") {
      isBarrel = false;
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
    }

    std::stringstream ss;
    auto print_path = [&](std::vector<std::pair<std::string, uint32_t>>& theHistory) {
      ss << " - ";
      for (const auto& t : theHistory) {
        ss << t.first;
        ss << "[";
        ss << t.second;
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
      print_path(geoHistory);

#ifdef EDM_ML_DEBUG
      edm::LogInfo("DD4hep_TestMTDIdealGeometry") << level << " " << clevel << " " << fv.name() << " " << ccopy << "\n"
                                                  << fv.path();
#endif

      edm::LogInfo("DD4hep_TestMTDPath") << ss.str();

      bool isSens = false;

      for (auto const& t : specs) {
        for (auto const& it : t->paths) {
          if (dd::compareEqual(fv.name(), dd::realTopName(it))) {
            isSens = true;
            break;
          }
        }
      }

      if (isSens) {
        //
        // Test of numbering scheme for sensitive detectors
        //

        theBaseNumber(geoHistory);

        std::stringstream snum;

        if (isBarrel) {
          BTLDetId::CrysLayout lay = static_cast<BTLDetId::CrysLayout>(theLayout_);
          BTLDetId theId(btlNS_.getUnitID(thisN_));
          int hIndex = theId.hashedIndex(lay);
          BTLDetId theNewId(theId.getUnhashedIndex(hIndex, lay));
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
          snum << theId;
        }
        edm::LogInfo("DD4hep_TestMTDNumbering") << snum.str();

        //
        // Test of positions for sensitive detectors
        //

        std::stringstream spos;

        if (!fv.isABox()) {
          throw cms::Exception("TestMTDIdealGeometry") << "MTD sensitive element not a DDBox";
          break;
        }
        dd::DDBox mySens(fv);
        spos << "Solid shape name: " << DDSolidShapesName::name(fv.legacyShape(dd::getCurrentShape(fv))) << "\n";
        spos << "Box dimensions: " << std::setw(14) << std::fixed << convertCmToMm(mySens.halfX()) << " "
             << std::setw(14) << convertCmToMm(mySens.halfY()) << " " << std::setw(14) << convertCmToMm(mySens.halfZ())
             << "\n";

        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        spos << "Translation vector components: " << std::setw(14) << std::fixed << convertCmToMm(fv.translation().x())
             << " " << std::setw(14) << convertCmToMm(fv.translation().y()) << " " << std::setw(14)
             << convertCmToMm(fv.translation().z()) << " "
             << "\n";
        spos << "Rotation matrix components: " << std::setw(14) << x.X() << " " << std::setw(14) << x.Y() << " "
             << std::setw(14) << x.Z() << " " << std::setw(14) << y.X() << " " << std::setw(14) << y.Y() << " "
             << std::setw(14) << y.Z() << " " << std::setw(14) << z.X() << " " << std::setw(14) << z.Y() << " "
             << std::setw(14) << z.Z() << "\n";

        DD3Vector zeroLocal(0., 0., 0.);
        DD3Vector cn1Local(mySens.halfX(), mySens.halfY(), mySens.halfZ());
        double distLocal = cn1Local.R();
        DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
        DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();
        double distGlobal =
            std::sqrt(std::pow(zeroGlobal.X() - cn1Global.X(), 2) + std::pow(zeroGlobal.Y() - cn1Global.Y(), 2) +
                      std::pow(zeroGlobal.Z() - cn1Global.Z(), 2));

        spos << "Center global   = " << std::setw(14) << convertCmToMm(zeroGlobal.X()) << std::setw(14)
             << convertCmToMm(zeroGlobal.Y()) << std::setw(14) << convertCmToMm(zeroGlobal.Z())
             << " r = " << std::setw(14) << convertCmToMm(zeroGlobal.Rho()) << " phi = " << std::setw(14)
             << convertRadToDeg(zeroGlobal.Phi()) << "\n";

        spos << "Corner 1 local  = " << std::setw(14) << convertCmToMm(cn1Local.X()) << std::setw(14)
             << convertCmToMm(cn1Local.Y()) << std::setw(14) << convertCmToMm(cn1Local.Z())
             << " DeltaR = " << std::setw(14) << convertCmToMm(distLocal) << "\n";

        spos << "Corner 1 global = " << std::setw(14) << convertCmToMm(cn1Global.X()) << std::setw(14)
             << convertCmToMm(cn1Global.Y()) << std::setw(14) << convertCmToMm(cn1Global.Z())
             << " DeltaR = " << std::setw(14) << convertCmToMm(distGlobal) << "\n";

        spos << "\n";
        if (std::fabs(convertCmToMm(distGlobal - distLocal)) > 1.e-6) {
          spos << "DIFFERENCE IN DISTANCE \n";
        }
        edm::LogInfo("DD4hep_TestMTDPosition") << spos.str();
      }
    }
  } while (fv.next(0));
}

void DD4hep_TestMTDIdealGeometry::theBaseNumber(const std::vector<std::pair<std::string, uint32_t>>& gh) {
  thisN_.reset();
  thisN_.setSize(gh.size());

  for (auto t = gh.rbegin(); t != gh.rend(); ++t) {
    std::string name;
    name.assign(t->first);
    int copyN(t->second);
    thisN_.addLevel(name, copyN);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_TestMTDIdealGeometry") << name << " " << copyN;
#endif
  }
}

DEFINE_FWK_MODULE(DD4hep_TestMTDIdealGeometry);
