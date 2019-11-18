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
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

//#define EDM_ML_DEBUG

using namespace cms;

class DD4hep_TestMTDNumbering : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_TestMTDNumbering(const edm::ParameterSet&);
  ~DD4hep_TestMTDNumbering() = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(const std::vector<std::pair<std::string_view, uint32_t>>& gh);

private:
  const edm::ESInputTag tag_;
  std::string fname_;
  std::string ddTopNodeName_;
  uint32_t theLayout_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
};

DD4hep_TestMTDNumbering::DD4hep_TestMTDNumbering(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")),
      fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
      ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "BarrelTimingLayer")),
      theLayout_(iConfig.getUntrackedParameter<uint32_t>("theLayout", 1)),
      thisN_(),
      btlNS_(),
      etlNS_() {
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
}

void DD4hep_TestMTDNumbering::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

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

  if (!pSP.isValid()) {
    edm::LogError("DD4hep_TestMTDNumbering") << "ESTransientHandle<DDSpecParRegistry> pSP is not valid!";
    return;
  }

  const std::string fname = "dump" + fname_;

  DDFilteredView fv(pDD.product(), pDD.product()->description()->worldVolume());
  fv.next(0);
  edm::LogInfo("DD4hep_TestMTDNumbering") << fv.name();

  DDSpecParRefs specs;
  std::string attribute("ReadOutName"), name;
  if (ddTopNodeName_ == "BarrelTimingLayer") {
    name = "FastTimerHitsBarrel";
  } else if (ddTopNodeName_ == "EndcapTimingLayer") {
    name = "FastTimerHitsEndcap";
  }
  if (name.empty()) {
    edm::LogError("DD4hep_TestMTDNumbering") << "No sensitive detector provided, abort";
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

  std::ofstream dump(fname.c_str());

  bool write = false;
  bool isBarrel = true;
  uint32_t level(0);
  std::vector<std::pair<std::string_view, uint32_t>> geoHistory;

  do {
    uint32_t clevel = fv.navPos().size();
    uint32_t ccopy = (clevel > 1 ? fv.copyNum() : 0);
    geoHistory.resize(clevel);
    geoHistory[clevel - 1] = std::pair<std::string_view, uint32_t>(fv.name(), ccopy);

    if (fv.name() == "BarrelTimingLayer") {
      isBarrel = true;
      edm::LogInfo("DD4hep_TestMTDNumbering") << "isBarrel = " << isBarrel;
    } else if (fv.name() == "EndcapTimingLayer") {
      isBarrel = false;
      edm::LogInfo("DD4hep_TestMTDNumbering") << "isBarrel = " << isBarrel;
    }

    auto print_path = [&](std::vector<std::pair<std::string_view, uint32_t>>& theHistory) {
      dump << " - ";
      for (const auto& t : theHistory) {
        dump << t.first + "[" + t.second + "]/";
      }
      dump << "\n";
    };

#ifdef EDM_ML_DEBUG
    edm::LogInfo("DD4hep_TestMTDNumbering") << level << " " << clevel << " " << fv.name() << " " << ccopy;
    edm::LogVerbatim("DD4hep_TestMTDNumbering").log([&geoHistory](auto& log) {
      for (const auto& t : geoHistory) {
        log << t.first + "[" + t.second + "]/";
      }
    });
#endif
    if (level > 0 && fv.navPos().size() < level) {
      level = 0;
      write = false;
    }
    if (fv.name() == ddTopNodeName_) {
      write = true;
      level = fv.navPos().size();
    }

    // Actions for MTD volumes: searchg for sensitive detectors

    if (write) {
      print_path(geoHistory);

      bool isSens = false;

      for (auto const& t : specs) {
        for (auto const& it : t->paths) {
          if (dd::compareEqual(fv.name(), dd::realTopName(it))) {
            isSens = true;
            break;
          }
        }
      }

      // Check of numbering scheme for sensitive detectors

      if (isSens) {
        theBaseNumber(geoHistory);

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
  } while (fv.next(0));
  dump << std::flush;
  dump.close();
}

void DD4hep_TestMTDNumbering::theBaseNumber(const std::vector<std::pair<std::string_view, uint32_t>>& gh) {
  thisN_.reset();
  thisN_.setSize(gh.size());

  for (auto t = gh.rbegin(); t != gh.rend(); ++t) {
    std::string name;
    name.assign(t->first);
    int copyN(t->second);
    thisN_.addLevel(name, copyN);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_TestMTDNumbering") << name << " " << copyN;
#endif
  }
}

DEFINE_FWK_MODULE(DD4hep_TestMTDNumbering);
