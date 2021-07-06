/****************************************************************************
 *
 * Authors:
 *  Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardMisalignedGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

#include <iostream>
#include <iomanip>

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to print out information on current geometry.
 **/
class CTPPSGeometryInfo : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSGeometryInfo(const edm::ParameterSet&);

private:
  const std::string geometryType_;
  const bool printRPInfo_, printSensorInfo_;
  const edm::ESGetToken<CTPPSGeometry, IdealGeometryRecord> tokIdeal_;
  const edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> tokReal_;
  const edm::ESGetToken<CTPPSGeometry, VeryForwardMisalignedGeometryRecord> tokMis_;

  edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
  edm::ESWatcher<VeryForwardRealGeometryRecord> watcherRealGeometry_;
  edm::ESWatcher<VeryForwardMisalignedGeometryRecord> watcherMisalignedGeometry_;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  static std::string formatDetId(const CTPPSDetId& id, bool printDetails = true);

  void printGeometry(const CTPPSGeometry&, const edm::Event&);
};

//----------------------------------------------------------------------------------------------------

CTPPSGeometryInfo::CTPPSGeometryInfo(const edm::ParameterSet& iConfig)
    : geometryType_(iConfig.getUntrackedParameter<std::string>("geometryType", "real")),
      printRPInfo_(iConfig.getUntrackedParameter<bool>("printRPInfo", true)),
      printSensorInfo_(iConfig.getUntrackedParameter<bool>("printSensorInfo", true)),
      tokIdeal_(esConsumes<CTPPSGeometry, IdealGeometryRecord>()),
      tokReal_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord>()),
      tokMis_(esConsumes<CTPPSGeometry, VeryForwardMisalignedGeometryRecord>()) {}

//----------------------------------------------------------------------------------------------------

void CTPPSGeometryInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (geometryType_ == "ideal") {
    if (watcherIdealGeometry_.check(iSetup)) {
      const auto& geometry = iSetup.getData(tokIdeal_);
      printGeometry(geometry, iEvent);
    }
    return;
  }

  else if (geometryType_ == "real") {
    if (watcherRealGeometry_.check(iSetup)) {
      const auto& geometry = iSetup.getData(tokReal_);
      printGeometry(geometry, iEvent);
    }
    return;
  }

  else if (geometryType_ == "misaligned") {
    if (watcherMisalignedGeometry_.check(iSetup)) {
      const auto& geometry = iSetup.getData(tokMis_);
      printGeometry(geometry, iEvent);
    }
    return;
  }

  throw cms::Exception("CTPPSGeometryInfo") << "Unknown geometry type: `" << geometryType_ << "'.";
}

//----------------------------------------------------------------------------------------------------

std::string CTPPSGeometryInfo::formatDetId(const CTPPSDetId& id, bool printDetails) {
  std::ostringstream oss;
  oss << id.rawId();

  const unsigned int rpDecId = id.arm() * 100 + id.station() * 10 + id.rp();

  if (id.subdetId() == CTPPSDetId::sdTrackingStrip) {
    TotemRPDetId fid(id);
    oss << " (strip RP " << std::setw(3) << rpDecId;
    if (printDetails)
      oss << ", plane " << fid.plane();
    oss << ")";
  }

  else if (id.subdetId() == CTPPSDetId::sdTrackingPixel) {
    CTPPSPixelDetId fid(id);
    oss << " (pixel RP " << std::setw(3) << rpDecId;
    if (printDetails)
      oss << ", plane " << fid.plane();
    oss << ")";
  }

  else if (id.subdetId() == CTPPSDetId::sdTimingDiamond) {
    CTPPSDiamondDetId fid(id);
    oss << " (diamd RP " << std::setw(3) << rpDecId;
    if (printDetails)
      oss << ", plane " << fid.plane() << ", channel " << std::setw(2) << fid.channel();
    oss << ")";
  }

  else if (id.subdetId() == CTPPSDetId::sdTimingFastSilicon) {
    TotemTimingDetId fid(id);
    oss << " (totim RP " << std::setw(3) << rpDecId;
    if (printDetails)
      oss << ", plane " << fid.plane() << ", channel " << std::setw(2) << fid.channel();
    oss << ")";
  }

  return oss.str();
}

//----------------------------------------------------------------------------------------------------

void CTPPSGeometryInfo::printGeometry(const CTPPSGeometry& geometry, const edm::Event& event) {
  time_t unixTime = event.time().unixTime();
  char timeStr[50];
  strftime(timeStr, 50, "%F %T", localtime(&unixTime));

  std::ostringstream oss;

  // RP geometry
  if (printRPInfo_) {
    oss << "* RPs:\n"
        << "    ce: RP center in global coordinates, in mm\n";

    for (auto it = geometry.beginRP(); it != geometry.endRP(); ++it) {
      const DetGeomDesc::Translation& t = it->second->translation();

      oss << formatDetId(CTPPSDetId(it->first), false) << std::fixed << std::setprecision(3) << std::showpos
          << " | ce=(" << t.x() << ", " << t.y() << ", " << t.z() << ")\n";
    }

    edm::LogVerbatim("CTPPSGeometryInfo") << oss.str();
  }

  // sensor geometry
  if (printSensorInfo_) {
    oss << "* sensors:\n"
        << "    ce: sensor center in global coordinates, in mm\n"
        << "    a1: local axis (1, 0, 0) in global coordinates\n"
        << "    a2: local axis (0, 1, 0) in global coordinates\n"
        << "    a3: local axis (0, 0, 1) in global coordinates\n";

    for (auto it = geometry.beginSensor(); it != geometry.endSensor(); ++it) {
      CTPPSDetId detId(it->first);

      const auto gl_o = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 0, 0));
      const auto gl_a1 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(1, 0, 0)) - gl_o;
      const auto gl_a2 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 1, 0)) - gl_o;
      const auto gl_a3 = geometry.localToGlobal(detId, CTPPSGeometry::Vector(0, 0, 1)) - gl_o;

      oss << formatDetId(detId) << std::fixed << std::setprecision(3) << std::showpos << " | ce=(" << gl_o.x() << ", "
          << gl_o.y() << ", " << gl_o.z() << ")"
          << " | a1=(" << gl_a1.x() << ", " << gl_a1.y() << ", " << gl_a1.z() << ")"
          << " | a2=(" << gl_a2.x() << ", " << gl_a2.y() << ", " << gl_a2.z() << ")"
          << " | a3=(" << gl_a3.x() << ", " << gl_a3.y() << ", " << gl_a3.z() << ")\n";
    }
  }

  edm::LogInfo("CTPPSGeometryInfo") << "New " << geometryType_ << " geometry found in run=" << event.id().run()
                                    << ", event=" << event.id().event() << ", UNIX timestamp=" << unixTime << " ("
                                    << timeStr << ")\n"
                                    << oss.str();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSGeometryInfo);
