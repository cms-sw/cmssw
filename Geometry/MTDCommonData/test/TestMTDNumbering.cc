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

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

//#define EDM_ML_DEBUG

class TestMTDNumbering : public edm::one::EDAnalyzer<> {
public:
  explicit TestMTDNumbering(const edm::ParameterSet&);
  ~TestMTDNumbering() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(const DDGeoHistory& gh);

  std::string noNSgeoHistory(const DDGeoHistory& gh);

private:
  std::string label_;
  std::string fname_;
  int nNodes_;
  std::string ddTopNodeName_;
  uint32_t theLayout_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;
};

TestMTDNumbering::TestMTDNumbering(const edm::ParameterSet& iConfig)
    : label_(iConfig.getUntrackedParameter<std::string>("label", "")),
      fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      theLayout_(iConfig.getUntrackedParameter<uint32_t>("theLayout", 1)),
      thisN_(),
      btlNS_(),
      etlNS_() {}

TestMTDNumbering::~TestMTDNumbering() {}

void TestMTDNumbering::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (ddTopNodeName_ != "BarrelTimingLayer" && ddTopNodeName_ != "EndcapTimingLayer") {
    edm::LogWarning("TestMTDNumbering") << ddTopNodeName_ << "Not valid top MTD volume";
    return;
  }

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(label_, pDD);

  if (!pDD.isValid()) {
    edm::LogError("TestMTDNumbering") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("TestMTDNumbering") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("TestMTDNumbering") << "NO label found pDD.description() returned false.";
  }

  std::string fname = "dump" + fname_;

  DDPassAllFilter filter;
  DDFilteredView fv(*pDD, filter);

  edm::LogInfo("TestMTDNumbering") << "Top Most LogicalPart = " << fv.logicalPart();

  using nav_type = DDFilteredView::nav_type;
  using id_type = std::map<nav_type, int>;
  id_type idMap;
  int id = 0;
  std::ofstream dump(fname.c_str());

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
      edm::LogInfo("TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    } else if (fv.geoHistory()[num - 1].logicalPart().name() == "etl:EndcapTimingLayer") {
      isBarrel = false;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    }

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
        theBaseNumber(fv.geoHistory());

        if (isBarrel) {
          BTLDetId::CrysLayout lay = static_cast<BTLDetId::CrysLayout>(theLayout_);
          BTLDetId theId(btlNS_.getUnitID(thisN_));
          int hIndex = theId.hashedIndex(lay);
          BTLDetId theNewId(theId.getUnhashedIndex(hIndex, lay));
          dump << theId;
          dump << "\n layout type = " << static_cast<int>(lay);
          dump << "\n ieta        = " << theId.ieta(lay);
          dump << "\n iphi        = " << theId.iphi(lay);
          dump << "\n hashedIndex = " << theId.hashedIndex(lay);
          dump << "\n BTLDetId hI = " << theNewId;
          if (theId.mtdSide() != theNewId.mtdSide()) {
            dump << "\n DIFFERENCE IN SIDE";
          }
          if (theId.mtdRR() != theNewId.mtdRR()) {
            dump << "\n DIFFERENCE IN ROD";
          }
          if (theId.module() != theNewId.module()) {
            dump << "\n DIFFERENCE IN MODULE";
          }
          if (theId.modType() != theNewId.modType()) {
            dump << "\n DIFFERENCE IN MODTYPE";
          }
          if (theId.crystal() != theNewId.crystal()) {
            dump << "\n DIFFERENCE IN CRYSTAL";
          }
          dump << "\n";
        } else {
          ETLDetId theId(etlNS_.getUnitID(thisN_));
          dump << theId;
        }
        dump << "\n";
      }
    }
    ++id;
  } while (fv.next());
  dump << std::flush;
  dump.close();
}

void TestMTDNumbering::theBaseNumber(const DDGeoHistory& gh) {
  thisN_.reset();
  thisN_.setSize(gh.size());

  for (uint i = gh.size(); i-- > 0;) {
    std::string name(gh[i].logicalPart().name().name());
    int copyN(gh[i].copyno());
    thisN_.addLevel(name, copyN);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("TestMTDNumbering") << name << " " << copyN;
#endif
  }
}

std::string TestMTDNumbering::noNSgeoHistory(const DDGeoHistory& gh) {
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

DEFINE_FWK_MODULE(TestMTDNumbering);
