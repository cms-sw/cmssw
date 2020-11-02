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
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "DetectorDescription/DDCMS/interface/DDDetector.h"
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

class DD4hep_MTDTopologyAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit DD4hep_MTDTopologyAnalyzer(const edm::ParameterSet&);
  ~DD4hep_MTDTopologyAnalyzer() = default;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber(cms::DDFilteredView& fv);

private:
  const edm::ESInputTag tag_;

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

  edm::ESGetToken<DDDetector, IdealGeometryRecord> dddetToken_;
  edm::ESGetToken<DDSpecParRegistry, DDSpecParRegistryRcd> dspecToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
};

DD4hep_MTDTopologyAnalyzer::DD4hep_MTDTopologyAnalyzer(const edm::ParameterSet& iConfig)
    : tag_(iConfig.getParameter<edm::ESInputTag>("DDDetector")), thisN_(), btlNS_(), etlNS_() {
  dddetToken_ = esConsumes<DDDetector, IdealGeometryRecord>(tag_);
  dspecToken_ = esConsumes<DDSpecParRegistry, DDSpecParRegistryRcd>(tag_);
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>(tag_);
}

void DD4hep_MTDTopologyAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pDD = iSetup.getTransientHandle(dddetToken_);

  auto pSP = iSetup.getTransientHandle(dspecToken_);

  if (!pDD.isValid()) {
    edm::LogError("MTDTopologyAnalyzer") << "ESTransientHandle<DDCompactView> pDD is not valid!";
    return;
  }
  if (pDD.description()) {
    edm::LogInfo("MTDTopologyAnalyzer") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("MTDTopologyAnalyzer") << "NO label found pDD.description() returned false.";
  }

  if (!pSP.isValid()) {
    edm::LogError("MTDTopologyAnalyzer") << "ESTransientHandle<DDSpecParRegistry> pSP is not valid!";
    return;
  }

  auto pTP = iSetup.getTransientHandle(mtdtopoToken_);
  if (!pTP.isValid()) {
    edm::LogError("MTDTopologyAnalyzer") << "ESTransientHandle<MTDTopology> pTP is not valid!";
    return;
  } else {
    edm::LogInfo("MTDTopologyAnalyzer") << "MTD topology mode = " << pTP->getMTDTopologyMode();
  }

  DDFilteredView fv(pDD.product(), pDD.product()->worldVolume());
  fv.next(0);
  edm::LogInfo("MTDTopologyAnalyzer") << fv.name();

  DDSpecParRefs specs;
  std::string attribute("ReadOutName"), name;
  //
  // Select both BTL and ETL sensitive volumes
  //
  name = "FastTimerHitsBarrel";
  pSP.product()->filter(specs, attribute, name);
  name = "FastTimerHitsEndcap";
  pSP.product()->filter(specs, attribute, name);

  edm::LogVerbatim("Geometry").log([&specs](auto& log) {
    log << "Filtered DD SpecPar Registry size: " << specs.size() << "\n";
    for (const auto& t : specs) {
      log << "\nRegExps { ";
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
  std::string ddtop("");
  uint32_t level(0);

  do {
    if (fv.name() == "BarrelTimingLayer") {
      isBarrel = true;
      ddtop = "BarrelTimingLayer";
      edm::LogInfo("MTDTopologyAnalyzer") << "isBarrel = " << isBarrel;
    } else if (fv.name() == "EndcapTimingLayer") {
      isBarrel = false;
      ddtop = "EndcapTimingLayer";
      edm::LogInfo("MTDTopologyAnalyzer") << "isBarrel = " << isBarrel;
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MTDTopologyAnalyzer") << "Top level volume: " << ddtop;
#endif

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

    if (level > 0 && fv.copyNos().size() < level) {
      level = 0;
      ddtop.clear();
    }
    if (fv.name() == "BarrelTimingLayer" || fv.name() == "EndcapTimingLayer") {
      level = fv.copyNos().size();
    }

    if (!ddtop.empty()) {
      // Actions for MTD topology test: searchg for sensitive detectors

      print_path();

#ifdef EDM_ML_DEBUG
      edm::LogInfo("MTDTopologyAnalyzer") << "Top level volume: " << ddtop << " at history " << ss.str();
#endif

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

        edm::LogVerbatim("MTDTopologyanalyzer") << ss.str();

        if (isBarrel) {
          BTLDetId theId(btlNS_.getUnitID(thisN_));
          DetId localId(theId.rawId());
          edm::LogVerbatim("MTDTopologAnalyzer") << pTP->print(localId) << "\n" << theId;
        } else {
          ETLDetId theId(etlNS_.getUnitID(thisN_));
          DetId localId(theId.rawId());
          edm::LogVerbatim("MTDTopologAnalyzer") << pTP->print(localId) << "\n" << theId;
        }
      }
    }
  } while (fv.next(0));
}

void DD4hep_MTDTopologyAnalyzer::theBaseNumber(cms::DDFilteredView& fv) {
  thisN_.reset();
  thisN_.setSize(fv.copyNos().size());

  for (uint ii = 0; ii < fv.copyNos().size(); ii++) {
    std::string name(dd4hep::dd::noNamespace((fv.geoHistory()[ii])->GetName()));
    name.assign(name.erase(name.rfind('_')));
    int copyN(fv.copyNos()[ii]);
    thisN_.addLevel(name, copyN);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MTDTopologyAnalyzer") << name << " " << copyN;
#endif
  }
}

DEFINE_FWK_MODULE(DD4hep_MTDTopologyAnalyzer);
