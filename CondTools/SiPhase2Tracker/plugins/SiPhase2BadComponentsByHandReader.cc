// -*- C++ -*-
//
// Package:    CondTools/SiPhase2BadComponentsByHandReader
// Class:      SiPhase2BadComponentsByHandReader
//
/**\class SiPhase2BadComponentsByHandReader SiPhase2BadComponentsByHandReader.cc CondTools/SiPhase2Tracker/plugins/SiPhase2BadComponentsByHandReader.cc

 Description: Reads back a SiPixelQuality disabled-module payload from the EventSetup and prints each DetId with its module type and any paired sensor. Templated over the record type; OT and IT instances are provided.

 Implementation:
     Verification tool for the builder output. The OT pairing uses a map built from StackGeomDet, and IT Ph2PXB3D pairs use DetId parity.
*/
//
// Original Author:  Stef Duponcheel
//         Created:  Thu, 08 Oct 2026 14:43:39 GMT
//
//

#include <map>
#include <set>
#include <string>

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerCondDataRecords.h"
#include "CondFormats/DataRecord/interface/SiPhase2InnerTrackerCondDataRecords.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/StackGeomDet.h"

//
// class declaration
//
template <typename RecordT>
class SiPhase2BadComponentsByHandReader : public edm::global::EDAnalyzer<> {
public:
  explicit SiPhase2BadComponentsByHandReader(const edm::ParameterSet& iConfig)
      : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", true)),
        geomToken_(esConsumes()),
        badModuleToken_(esConsumes(edm::ESInputTag{"", iConfig.getUntrackedParameter<std::string>("label", "")})) {}

  ~SiPhase2BadComponentsByHandReader() override = default;
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  static std::string moduleTypeToString(TrackerGeometry::ModuleType type);

  const bool printdebug_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<SiPixelQuality, RecordT> badModuleToken_;
};

template <typename RecordT>
std::string SiPhase2BadComponentsByHandReader<RecordT>::moduleTypeToString(TrackerGeometry::ModuleType type) {
  switch (type) {
    case TrackerGeometry::ModuleType::Ph2SS:
      return "Ph2SS";
    case TrackerGeometry::ModuleType::Ph2PSP:
      return "Ph2PSP";
    case TrackerGeometry::ModuleType::Ph2PSS:
      return "Ph2PSS";
    case TrackerGeometry::ModuleType::Ph2PXB:
      return "Ph2PXB";
    case TrackerGeometry::ModuleType::Ph2PXF:
      return "Ph2PXF";
    case TrackerGeometry::ModuleType::Ph2PXB3D:
      return "Ph2PXB3D";
    default:
      return "unknown";
  }
}

template <typename RecordT>
void SiPhase2BadComponentsByHandReader<RecordT>::analyze(edm::StreamID,
                                                         edm::Event const&,
                                                         edm::EventSetup const& iSetup) const {
  const auto& tkGeom = iSetup.getData(geomToken_);
  const auto& payload = iSetup.getData(badModuleToken_);

  if (!printdebug_) {
    return;
  }

  // Find the ID's in the OT stack
  std::map<uint32_t, uint32_t> otPartnerOf;
  for (auto const* det : tkGeom.dets()) {
    if (const StackGeomDet* stack = dynamic_cast<const StackGeomDet*>(det)) {
      uint32_t lowerId = stack->lowerDet()->geographicalId().rawId();
      uint32_t upperId = stack->upperDet()->geographicalId().rawId();
      otPartnerOf[lowerId] = upperId;
      otPartnerOf[upperId] = lowerId;
    }
  }

  std::set<uint32_t> allDetIds;
  for (const auto& bc : payload.getBadComponentList()) {
    allDetIds.insert(bc.DetID);
  }

  for (const auto& bc : payload.getBadComponentList()) {
    uint32_t id = bc.DetID;
    TrackerGeometry::ModuleType type = tkGeom.getDetectorType(DetId(id));
    std::string pairInfo;

    if (type == TrackerGeometry::ModuleType::Ph2PXB3D) {
      uint32_t partner = (id % 2 == 1) ? (id + 1) : (id - 1);
      if (allDetIds.count(partner)) {
        pairInfo = ", paired with " + std::to_string(partner);
      }
    } else {
      auto it = otPartnerOf.find(id);
      if (it != otPartnerOf.end() && allDetIds.count(it->second)) {
        pairInfo = ", paired with " + std::to_string(it->second);
      }
    }

    edm::LogInfo("SiPhase2BadComponentsByHandReader")
        << "Disabled module DetId " << id << " (type: " << moduleTypeToString(type) << ")" << pairInfo;
  }
}

template <typename RecordT>
void SiPhase2BadComponentsByHandReader<RecordT>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("printDebug", true);
  desc.addUntracked<std::string>("label", "");
  descriptions.addWithDefaultLabel(desc);
}

using SiPhase2OTBadComponentsReader = SiPhase2BadComponentsByHandReader<SiPhase2OuterTrackerBadModuleRcd>;
using SiPhase2ITBadComponentsReader = SiPhase2BadComponentsByHandReader<SiPhase2InnerTrackerBadModuleRcd>;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiPhase2OTBadComponentsReader);
DEFINE_FWK_MODULE(SiPhase2ITBadComponentsReader);
