#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

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

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/Math/interface/Rounding.h"

//#define EDM_ML_DEBUG

class TestMTDIdealGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit TestMTDIdealGeometry(const edm::ParameterSet&);
  ~TestMTDIdealGeometry() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(const DDGeoHistory& gh);

private:
  std::string label_;
  int nNodes_;
  std::string ddTopNodeName_;
  uint32_t theLayout_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;
};

TestMTDIdealGeometry::TestMTDIdealGeometry(const edm::ParameterSet& iConfig)
    : label_(iConfig.getUntrackedParameter<std::string>("label", "")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      theLayout_(iConfig.getUntrackedParameter<uint32_t>("theLayout", 1)),
      thisN_(),
      btlNS_(),
      etlNS_() {}

TestMTDIdealGeometry::~TestMTDIdealGeometry() {}

using angle_units::operators::convertRadToDeg;
using cms_rounding::roundIfNear0;

void TestMTDIdealGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (ddTopNodeName_ != "BarrelTimingLayer" && ddTopNodeName_ != "EndcapTimingLayer") {
    edm::LogWarning("TestMTDIdealGeometry") << ddTopNodeName_ << "Not valid top MTD volume";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(label_, pDD);

  if (!pDD.isValid()) {
    edm::LogError("TestMTDIdealGeometry") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("TestMTDIdealGeometry") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("TestMTDIdealGeometry") << "NO label found pDD.description() returned false.";
  }

  DDPassAllFilter filter;
  DDFilteredView fv(*pDD, filter);

  edm::LogInfo("TestMTDIdealGeometry") << "Top Most LogicalPart = " << fv.logicalPart();

  using nav_type = DDFilteredView::nav_type;
  using id_type = std::map<nav_type, int>;
  id_type idMap;
  int id = 0;

  bool write = false;
  bool isBarrel = true;
  size_t limit = 0;

  do {
    nav_type pos = fv.navPos();
    idMap[pos] = id;

    size_t num = fv.geoHistory().size();

    if (num <= limit) {
      write = false;
    }
    if (fv.geoHistory()[num - 1].logicalPart().name() == "btl:BarrelTimingLayer") {
      isBarrel = true;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
#endif
    } else if (fv.geoHistory()[num - 1].logicalPart().name() == "etl:EndcapTimingLayer") {
      isBarrel = false;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDIdealGeometry") << "isBarrel = " << isBarrel;
#endif
    }

    // Actions for MTD volumes: searchg for sensitive detectors

    std::stringstream ss;
    auto print_path = [&]() {
      ss << " - OCMS[0]/";
      for (uint i = 1; i < fv.geoHistory().size(); i++) {
        ss << fv.geoHistory()[i].logicalPart().name().fullname();
        ss << "[";
        ss << std::to_string(fv.geoHistory()[i].copyno());
        ss << "]/";
      }
    };

    if (write && fv.geoHistory()[limit - 1].logicalPart().name().name() == ddTopNodeName_) {
      print_path();
      edm::LogInfo("TestMTDPath") << ss.str();

      bool isSens = false;

      if (fv.geoHistory()[num - 1].logicalPart().specifics().size() > 0) {
        for (auto vec : fv.geoHistory()[num - 1].logicalPart().specifics()) {
          for (auto elem : *vec) {
            if (elem.second.name() == "SensitiveDetector") {
              isSens = true;
              break;
            }
          }
        }
      }

      if (isSens) {
        theBaseNumber(fv.geoHistory());

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
          snum << theId;
          snum << "\n layout type = " << static_cast<int>(lay);
          snum << "\n ieta        = " << theId.ieta(lay);
          snum << "\n iphi        = " << theId.iphi(lay);
          snum << "\n hashedIndex = " << theId.hashedIndex(lay);
          snum << "\n BTLDetId hI = " << theNewId;
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
#ifdef EDM_ML_DEBUG
          edm::LogInfo("TestMTDNumbering")
              << " ETLDetId = " << theId << "\n geographicalId = " << theId.geographicalId();
#endif
        }
        edm::LogInfo("TestMTDNumbering") << snum.str();

        //
        // Test of positions for sensitive detectors
        //

        std::stringstream spos;

        auto fround = [&](double in) {
          std::stringstream ss;
          ss << std::fixed << std::setw(14) << roundIfNear0(in);
          return ss.str();
        };

        DDBox mySens = fv.geoHistory()[num - 1].logicalPart().solid();
        spos << "Solid shape name: " << DDSolidShapesName::name(mySens.shape()) << "\n";
        if (static_cast<int>(mySens.shape()) != 1) {
          throw cms::Exception("TestMTDPosition") << "MTD sensitive element not a DDBox";
          break;
        }
        spos << "Box dimensions: " << fround(mySens.halfX()) << " " << fround(mySens.halfY()) << " "
             << fround(mySens.halfZ()) << "\n";

        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        spos << "Translation vector components: " << fround(fv.translation().x()) << " " << fround(fv.translation().y())
             << " " << fround(fv.translation().z()) << " "
             << "\n";
        spos << "Rotation matrix components: " << fround(x.X()) << " " << fround(x.Y()) << " " << fround(x.Z()) << " "
             << fround(y.X()) << " " << fround(y.Y()) << " " << fround(y.Z()) << " " << fround(z.X()) << " "
             << fround(z.Y()) << " " << fround(z.Z()) << "\n";

        DD3Vector zeroLocal(0., 0., 0.);
        DD3Vector cn1Local(mySens.halfX(), mySens.halfY(), mySens.halfZ());
        double distLocal = cn1Local.R();
        DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
        DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();
        ;
        double distGlobal =
            std::sqrt(std::pow(zeroGlobal.X() - cn1Global.X(), 2) + std::pow(zeroGlobal.Y() - cn1Global.Y(), 2) +
                      std::pow(zeroGlobal.Z() - cn1Global.Z(), 2));

        spos << "Center global   = " << fround(zeroGlobal.X()) << fround(zeroGlobal.Y()) << fround(zeroGlobal.Z())
             << " r = " << fround(zeroGlobal.Rho()) << " phi = " << fround(convertRadToDeg(zeroGlobal.Phi())) << "\n";

        spos << "Corner 1 local  = " << fround(cn1Local.X()) << fround(cn1Local.Y()) << fround(cn1Local.Z())
             << " DeltaR = " << fround(distLocal) << "\n";

        spos << "Corner 1 global = " << fround(cn1Global.X()) << fround(cn1Global.Y()) << fround(cn1Global.Z())
             << " DeltaR = " << fround(distGlobal) << "\n";

        spos << "\n";
        if (std::fabs(distGlobal - distLocal) > 1.e-6) {
          spos << "DIFFERENCE IN DISTANCE \n";
        }
        sunitt << fround(zeroGlobal.X()) << fround(zeroGlobal.Y()) << fround(zeroGlobal.Z()) << fround(cn1Global.X())
               << fround(cn1Global.Y()) << fround(cn1Global.Z());
        edm::LogInfo("TestMTDPosition") << spos.str();

        edm::LogVerbatim("MTDUnitTest") << sunitt.str();
      }
    }
    ++id;
  } while (fv.next());
}

void TestMTDIdealGeometry::theBaseNumber(const DDGeoHistory& gh) {
  thisN_.reset();
  thisN_.setSize(gh.size());

  for (uint i = gh.size(); i-- > 0;) {
    std::string name(gh[i].logicalPart().name().name());
    int copyN(gh[i].copyno());
    thisN_.addLevel(name, copyN);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("TestMTDIdealGeometry") << name << " " << copyN;
#endif
  }
}

DEFINE_FWK_MODULE(TestMTDIdealGeometry);
