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

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#define EDM_ML_DEBUG

using namespace cms;

class DD4hep_TestMTDNumbering : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestMTDNumbering(const edm::ParameterSet&);
  ~DD4hep_TestMTDNumbering() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  // void theBaseNumber(const DDGeoHistory& gh);

  void checkMTD(const DDCompactView& cpv,
                std::string fname = "GeoHistory",
                int nVols = 0,
                std::string ddtop_ = "mtd:BarrelTimingLayer");

private:
  const edm::ESInputTag tag_;
  std::string fname_;
  int nNodes_;
  std::string ddTopNodeName_;
  uint32_t theLayout_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;
};

DD4hep_TestMTDNumbering::DD4hep_TestMTDNumbering(const edm::ParameterSet& iConfig)
  : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
    fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
    nNodes_(iConfig.getUntrackedParameter<uint32_t>("numNodesToDump", 0)),
    ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "btl:BarrelTimingLayer")),
    theLayout_(iConfig.getUntrackedParameter<uint32_t>("theLayout", 1)),
    thisN_(),
    btlNS_(),
    etlNS_() {}

DD4hep_TestMTDNumbering::~DD4hep_TestMTDNumbering() {}

void DD4hep_TestMTDNumbering::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(tag_, pDD);

  if (ddTopNodeName_ != "BarrelTimingLayer" && ddTopNodeName_ != "EndcapTimingLayer") {
    edm::LogWarning("DD4hep_TestMTDNumbering") << ddTopNodeName_ << "Not valid top MTD volume";
    return;
  }

  if (!pDD.isValid()) {
    edm::LogError("DD4hep_TestMTDNumbering") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("DD4hep_TestMTDNumbering") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("DD4hep_TestMTDNumbering") << "NO label found pDD.description() returned false.";
  }

  checkMTD(*pDD, fname_, nNodes_, ddTopNodeName_);
}

void DD4hep_TestMTDNumbering::checkMTD(const DDCompactView& cpv, std::string fname, int nVols, std::string ddtop_) {
  fname = "dump" + fname;

  DDFilteredView fv(cpv);
  fv.next(0);
  edm::LogInfo("DD4hep_TestMTDNumbering") << fv.name();

  std::ofstream dump(fname.c_str());
  bool notReachedDepth(true);

  bool write = false;
  bool isBarrel = true;
  unsigned int level(0);
  size_t maxLevel = 50;
  std::vector<std::string> theLevels(maxLevel,"");

  do {

    unsigned int clevel = fv.navPos().size();
    unsigned int ccopy = (clevel > 1 ? fv.copyNum() : 0);
    theLevels[clevel-1] = fv.name()+"["+ccopy+"]/";

    if (fv.name() == "BarrelTimingLayer") {
      isBarrel = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("DD4hep_TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    } else if (fv.name() == "EndcapTimingLayer") {
      isBarrel = false;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("DD4hep_TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    }
    if ( level > 0 && fv.navPos().size() <= level ) { break; }
    if ( fv.name() == ddtop_ ) { write = true; level = fv.navPos().size(); }

    // Actions for MTD volumes: searchg for sensitive detectors

    if (write) {
      std::string thePath;
      for ( size_t st = 0; st < clevel; st++ ) {
        thePath += theLevels[st];
      }
      dump << " - " << thePath << "\n";

//       bool isSens = false;

//       if (fv.geoHistory()[num - 1].logicalPart().specifics().size() > 0) {
//         for (auto elem : *(fv.geoHistory()[num - 1].logicalPart().specifics()[0])) {
//           if (elem.second.name() == "SensitiveDetector") {
//             isSens = true;
//             break;
//           }
//         }
//       }

//       // Check of numbering scheme for sensitive detectors

//       if (isSens) {
//         theBaseNumber(fv.geoHistory());

//         if (isBarrel) {
//           BTLDetId::CrysLayout lay = static_cast<BTLDetId::CrysLayout>(theLayout_);
//           BTLDetId theId(btlNS_.getUnitID(thisN_));
//           int hIndex = theId.hashedIndex(lay);
//           BTLDetId theNewId(theId.getUnhashedIndex(hIndex, lay));
//           dump << theId;
//           dump << "\n layout type = " << static_cast<int>(lay);
//           dump << "\n ieta        = " << theId.ieta(lay);
//           dump << "\n iphi        = " << theId.iphi(lay);
//           dump << "\n hashedIndex = " << theId.hashedIndex(lay);
//           dump << "\n BTLDetId hI = " << theNewId;
//           if (theId.mtdSide() != theNewId.mtdSide()) {
//             dump << "\n DIFFERENCE IN SIDE";
//           }
//           if (theId.mtdRR() != theNewId.mtdRR()) {
//             dump << "\n DIFFERENCE IN ROD";
//           }
//           if (theId.module() != theNewId.module()) {
//             dump << "\n DIFFERENCE IN MODULE";
//           }
//           if (theId.modType() != theNewId.modType()) {
//             dump << "\n DIFFERENCE IN MODTYPE";
//           }
//           if (theId.crystal() != theNewId.crystal()) {
//             dump << "\n DIFFERENCE IN CRYSTAL";
//           }
//           dump << "\n";
//         } else {
//           ETLDetId theId(etlNS_.getUnitID(thisN_));
//           dump << theId;
//         }
//      dump << "\n";
//       }
    }
//     ++id;
//     if (nVols != 0 && id > nVols)
//       notReachedDepth = false;
  } while (fv.next(0) && notReachedDepth);
   dump << std::flush;
   dump.close();
}

// void DD4hep_TestMTDNumbering::theBaseNumber(const DDGeoHistory& gh) {
//   thisN_.reset();
//   thisN_.setSize(gh.size());

//   for (uint i = gh.size(); i-- > 0;) {
//     std::string name(gh[i].logicalPart().name().name());
//     int copyN(gh[i].copyno());
//     thisN_.addLevel(name, copyN);
// #ifdef EDM_ML_DEBUG
//     edm::LogInfo("DD4hep_TestMTDNumbering") << name << " " << copyN;
// #endif
//   }
// }

DEFINE_FWK_MODULE(DD4hep_TestMTDNumbering);
