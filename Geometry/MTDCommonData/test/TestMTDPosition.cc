#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

//#define EDM_ML_DEBUG

class TestMTDPosition : public edm::one::EDAnalyzer<> {
public:
  explicit TestMTDPosition(const edm::ParameterSet&);
  ~TestMTDPosition() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  std::string noNSgeoHistory(const DDGeoHistory& gh);

private:
  std::string label_;
  bool isMagField_;
  std::string fname_;
  int nNodes_;
  std::string ddTopNodeName_;

  static constexpr double rad2deg = 180. / M_PI;
};

TestMTDPosition::TestMTDPosition(const edm::ParameterSet& iConfig)
    : label_(iConfig.getUntrackedParameter<std::string>("label", "")),
      isMagField_(iConfig.getUntrackedParameter<bool>("isMagField", false)),
      fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
      nNodes_(iConfig.getUntrackedParameter<uint32_t>("numNodesToDump", 0)),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "btl:BarrelTimingLayer")) {
  if (isMagField_) {
    label_ = "magfield";
  }
}

TestMTDPosition::~TestMTDPosition() {}

void TestMTDPosition::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(label_, pDD);

  if (ddTopNodeName_ != "BarrelTimingLayer" && ddTopNodeName_ != "EndcapTimingLayer") {
    edm::LogWarning("TestMTDPosition") << ddTopNodeName_ << "Not valid top MTD volume";
    return;
  }

  if (!pDD.isValid()) {
    edm::LogError("TestMTDPosition") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("TestMTDPosition") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("TestMTDPosition") << "NO label found pDD.description() returned false.";
  }
  if (!pDD.isValid()) {
    edm::LogError("TestMTDPosition") << "ESTransientHandle<DDCompactView> pDD is not valid!";
  }

  std::string fname = "dump" + fname_;

  DDPassAllFilter filter;
  DDFilteredView fv(*pDD, filter);

  edm::LogInfo("TestMTDPosition") << "Top Most LogicalPart = " << fv.logicalPart();

  using nav_type = DDFilteredView::nav_type;
  using id_type = std::map<nav_type, int>;
  id_type idMap;
  int id = 0;
  std::ofstream dump(fname.c_str());

  bool write = false;
  size_t limit = 0;

  do {
    nav_type pos = fv.navPos();
    idMap[pos] = id;

    size_t num = fv.geoHistory().size();

    if (num <= limit) {
      write = false;
    }
    if (fv.geoHistory()[num - 1].logicalPart().name() == "btl:BarrelTimingLayer" ||
        fv.geoHistory()[num - 1].logicalPart().name() == "etl:EndcapTimingLayer") {
      limit = num;
      write = true;
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("TestMTDPosition") << fv.geoHistory();
    edm::LogVerbatim("TestMTDPosition") << "Translation = " << fv.translation().x() << " " << fv.translation().y()
                                        << " " << fv.translation().z();
#endif

    // Actions for MTD volumes: searchg for sensitive detectors

    if (write && fv.geoHistory()[limit - 1].logicalPart().name().name() == ddTopNodeName_) {
      dump << " - " << noNSgeoHistory(fv.geoHistory());
      dump << "\n";

      bool isSens = false;

      if (fv.geoHistory()[num - 1].logicalPart().specifics().size() > 0) {
        for (auto elem : *(fv.geoHistory()[num - 1].logicalPart().specifics()[0])) {
          if (elem.second.name() == "SensitiveDetector") {
            isSens = true;
            break;
          }
        }
      }

      // Check of numbering scheme for sensitive detectors

      if (isSens) {
        DDBox mySens = fv.geoHistory()[num - 1].logicalPart().solid();
        dump << "Solid shape name: " << DDSolidShapesName::name(mySens.shape()) << "\n";
        if (static_cast<int>(mySens.shape()) != 1) {
          throw cms::Exception("TestMTDPosition") << "MTD sensitive element not a DDBox";
          break;
        }
        dump << "Box dimensions: " << mySens.halfX() << " " << mySens.halfY() << " " << mySens.halfZ() << "\n";

        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        dump << "Translation vector components: " << std::setw(14) << std::fixed << fv.translation().x() << " "
             << std::setw(14) << fv.translation().y() << " " << std::setw(14) << fv.translation().z() << " "
             << "\n";
        dump << "Rotation matrix components: " << std::setw(14) << x.X() << " " << std::setw(14) << y.X() << " "
             << std::setw(14) << z.X() << " " << std::setw(14) << x.Y() << " " << std::setw(14) << y.Y() << " "
             << std::setw(14) << z.Y() << " " << std::setw(14) << x.Z() << " " << std::setw(14) << y.Z() << " "
             << std::setw(14) << z.Z() << " "
             << "\n";

        DD3Vector zeroLocal(0., 0., 0.);
        DD3Vector cn1Local(mySens.halfX(), mySens.halfY(), mySens.halfZ());
        double distLocal = cn1Local.R();
        DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
        DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();
        ;
        double distGlobal =
            std::sqrt(std::pow(zeroGlobal.X() - cn1Global.X(), 2) + std::pow(zeroGlobal.Y() - cn1Global.Y(), 2) +
                      std::pow(zeroGlobal.Z() - cn1Global.Z(), 2));

        dump << "Center global   = " << std::setw(14) << zeroGlobal.X() << std::setw(14) << zeroGlobal.Y()
             << std::setw(14) << zeroGlobal.Z() << " r = " << std::setw(14) << zeroGlobal.Rho()
             << " phi = " << std::setw(14) << zeroGlobal.Phi() * rad2deg << "\n";

        dump << "Corner 1 local  = " << std::setw(14) << cn1Local.X() << std::setw(14) << cn1Local.Y() << std::setw(14)
             << cn1Local.Z() << " DeltaR = " << std::setw(14) << distLocal << "\n";

        dump << "Corner 1 global = " << std::setw(14) << cn1Global.X() << std::setw(14) << cn1Global.Y()
             << std::setw(14) << cn1Global.Z() << " DeltaR = " << std::setw(14) << distGlobal << "\n";

        dump << "\n";
        if (std::fabs(distGlobal - distLocal) > 1.e-6) {
          dump << "DIFFERENCE IN DISTANCE \n";
        }
      }
    }
    ++id;
  } while (fv.next());
  dump << std::flush;
  dump.close();
}

std::string TestMTDPosition::noNSgeoHistory(const DDGeoHistory& gh) {
  std::string output;
  for (uint i = 0; i < gh.size(); i++) {
    output += gh[i].logicalPart().name().name();
    output += "[";
    output += std::to_string(gh[i].copyno());
    output += "]/";
  }

#ifdef EDM_ML_DEBUG
  edm::LogInfo("TestMTDNumbering") << output;
#endif

  return output;
}

DEFINE_FWK_MODULE(TestMTDPosition);
