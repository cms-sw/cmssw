// -*- C++ -*-
//
// Package:    CondTools/SiPhase2BadSensorsByHandReader
// Class:      SiPhase2BadSensorsByHandReader
//
/**\class SiPhase2BadSensorsByHandReader SiPhase2BadSensorsByHandReader.cc CondTools/SiPhase2Tracker/plugins/SiPhase2BadSensorsByHandReader.cc

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
class SiPhase2BadSensorsByHandReader : public edm::global::EDAnalyzer<> {
public:
  explicit SiPhase2BadSensorsByHandReader(const edm::ParameterSet& iConfig)
      : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug")),
        geomToken_(esConsumes()),
        badModuleToken_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("label")})) {}

  ~SiPhase2BadSensorsByHandReader() override = default;
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  static std::string moduleTypeToString(TrackerGeometry::ModuleType type);

  const bool printdebug_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<SiPixelQuality, RecordT> badModuleToken_;
};

template <typename RecordT>
std::string SiPhase2BadSensorsByHandReader<RecordT>::moduleTypeToString(TrackerGeometry::ModuleType type) {
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
void SiPhase2BadSensorsByHandReader<RecordT>::analyze(edm::StreamID,
                                                      edm::Event const&,
                                                      edm::EventSetup const& iSetup) const {
  const auto& tkGeom = iSetup.getData(geomToken_);
  const auto& payload = iSetup.getData(badModuleToken_);

  edm::LogInfo("SiPhase2BadSensorsByHandReader")
      << "Read " << payload.getBadComponentList().size() << " disabled components.";

  if (printdebug_) {
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

    std::set<uint32_t> alreadyReported;
    for (const auto& bc : payload.getBadComponentList()) {
      uint32_t id = bc.DetID;
      if (alreadyReported.count(id)) {
        continue;  // already logged together with its partner below
      }

      TrackerGeometry::ModuleType type = tkGeom.getDetectorType(DetId(id));
      std::string typeInfo = " (type: " + moduleTypeToString(type) + ")";

      uint32_t partner = 0;
      bool hasPartner = false;
      if (type == TrackerGeometry::ModuleType::Ph2PXB3D) {
        partner = (id % 2 == 1) ? (id + 1) : (id - 1);
        hasPartner = allDetIds.count(partner);
      } else {
        auto it = otPartnerOf.find(id);
        hasPartner = (it != otPartnerOf.end()) && allDetIds.count(it->second);
        partner = hasPartner ? it->second : 0;
      }

      if (hasPartner) {
        edm::LogInfo("SiPhase2BadSensorsByHandReader")
            << "Disabled module" << typeInfo << ": sensors " << id << " and " << partner;
        alreadyReported.insert(partner);
      } else {
        edm::LogInfo("SiPhase2BadSensorsByHandReader") << "Disabled sensor " << id << typeInfo;
      }
    }
  }
}

template <typename RecordT>
void SiPhase2BadSensorsByHandReader<RecordT>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("printDebug", false);
  desc.add<std::string>("label", "");
  descriptions.addWithDefaultLabel(desc);
}

using SiPhase2OTBadSensorsReader = SiPhase2BadSensorsByHandReader<SiPhase2OuterTrackerBadSensorRcd>;
using SiPhase2ITBadSensorsReader = SiPhase2BadSensorsByHandReader<SiPhase2InnerTrackerBadSensorRcd>;

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiPhase2OTBadSensorsReader);
DEFINE_FWK_MODULE(SiPhase2ITBadSensorsReader);
