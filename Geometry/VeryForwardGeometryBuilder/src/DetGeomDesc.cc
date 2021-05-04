/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*	Jan Kašpar (jan.kaspar@gmail.com)
*	CMSSW developers (based on GeometricDet class)
*
****************************************************************************/

#include <utility>

#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include <DD4hep/DD4hepUnits.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*
 *  Constructor from old DD DDFilteredView, also using the SpecPars to access 2x2 wafers info.
 */
DetGeomDesc::DetGeomDesc(const DDFilteredView& fv, const bool isRun2)
    : m_name(computeNameWithNoNamespace(fv.name())),
      m_copy(fv.copyno()),
      m_isDD4hep(false),
      m_trans(fv.translation()),  // mm (legacy)
      m_rot(fv.rotation()),
      m_params(fv.parameters()),  // default unit from old DD (mm)
      m_isABox(fv.shape() == DDSolidShape::ddbox),
      m_diamondBoxParams(computeDiamondDimensions(m_isABox, m_isDD4hep, m_params)),  // mm (legacy)
      m_sensorType(computeSensorType(fv.logicalPart().name().fullname())),
      m_geographicalID(computeDetID(m_name, fv.copyNumbers(), fv.copyno(), isRun2)),
      m_z(fv.translation().z())  // mm (legacy)
{}

/*
 *  Constructor from DD4Hep DDFilteredView, also using the SpecPars to access 2x2 wafers info.
 */
DetGeomDesc::DetGeomDesc(const cms::DDFilteredView& fv, const bool isRun2)
    : m_name(computeNameWithNoNamespace(fv.name())),
      m_copy(fv.copyNum()),
      m_isDD4hep(true),
      m_trans(fv.translation() / dd4hep::mm),  // converted from DD4hep unit to mm
      m_rot(fv.rotation()),
      m_params(computeParameters(fv)),  // default unit from DD4hep
      m_isABox(dd4hep::isA<dd4hep::Box>(fv.solid())),
      m_diamondBoxParams(computeDiamondDimensions(m_isABox, m_isDD4hep, m_params)),  // converted from DD4hep unit to mm
      m_sensorType(computeSensorType(fv.name())),
      m_geographicalID(computeDetIDFromDD4hep(m_name, fv.copyNos(), fv.copyNum(), isRun2)),
      m_z(fv.translation().z() / dd4hep::mm)  // converted from DD4hep unit to mm
{}

DetGeomDesc::DetGeomDesc(const DetGeomDesc& ref, CopyMode cm) {
  m_name = ref.m_name;
  m_copy = ref.m_copy;
  m_isDD4hep = ref.m_isDD4hep;
  m_trans = ref.m_trans;
  m_rot = ref.m_rot;
  m_params = ref.m_params;
  m_isABox = ref.m_isABox;
  m_diamondBoxParams = ref.m_diamondBoxParams;
  m_sensorType = ref.m_sensorType;
  m_geographicalID = ref.m_geographicalID;

  if (cm == cmWithChildren)
    m_container = ref.m_container;

  m_z = ref.m_z;
}

DetGeomDesc::~DetGeomDesc() { deepDeleteComponents(); }

void DetGeomDesc::addComponent(DetGeomDesc* det) { m_container.emplace_back(det); }

void DetGeomDesc::applyAlignment(const CTPPSRPAlignmentCorrectionData& t) {
  m_rot = t.getRotationMatrix() * m_rot;
  m_trans = t.getTranslation() + m_trans;
}

void DetGeomDesc::print() const {
  edm::LogVerbatim("DetGeomDesc::print") << "............................." << std::endl;
  edm::LogVerbatim("DetGeomDesc::print") << "name = " << m_name << std::endl;
  edm::LogVerbatim("DetGeomDesc::print") << "copy = " << m_copy << std::endl;
  edm::LogVerbatim("DetGeomDesc::print") << "translation = " << std::fixed << std::setprecision(7) << m_trans
                                         << std::endl;
  edm::LogVerbatim("DetGeomDesc::print") << "rotation = " << std::fixed << std::setprecision(7) << m_rot << std::endl;

  if (m_isABox) {
    edm::LogVerbatim("DetGeomDesc::print")
        << "getDiamondDimensions() = " << std::fixed << std::setprecision(7) << getDiamondDimensions().xHalfWidth << " "
        << getDiamondDimensions().yHalfWidth << " " << getDiamondDimensions().zHalfWidth << std::endl;
  }

  edm::LogVerbatim("DetGeomDesc::print") << "sensorType = " << m_sensorType << std::endl;

  if (m_geographicalID() != 0) {
    edm::LogVerbatim("DetGeomDesc::print") << "geographicalID() = " << m_geographicalID << std::endl;
  }

  edm::LogVerbatim("DetGeomDesc::print") << "parentZPosition() = " << std::fixed << std::setprecision(7) << m_z
                                         << std::endl;
}

/*
 * PRIVATE FUNCTIONS
 */

void DetGeomDesc::deleteComponents() { m_container.erase(m_container.begin(), m_container.end()); }

void DetGeomDesc::deepDeleteComponents() {
  for (auto& it : m_container) {
    delete it;  // the destructor calls deepDeleteComponents
  }
  clearComponents();
}

std::string DetGeomDesc::computeNameWithNoNamespace(std::string_view nameFromView) const {
  const auto& semiColonPos = nameFromView.find(":");
  const std::string name{(semiColonPos != std::string::npos ? nameFromView.substr(semiColonPos + 1) : nameFromView)};
  return name;
}

/*
 * Compute DD4hep shape parameters.
 */
std::vector<double> DetGeomDesc::computeParameters(const cms::DDFilteredView& fv) const {
  auto myShape = fv.solid();
  const std::vector<double>& parameters = myShape.dimensions();  // default unit from DD4hep (cm)
  return parameters;
}

/*
 * Compute diamond dimensions.
 * The diamond sensors are represented by the Box shape parameters.
 * oldDD: params are already in mm.
 * DD4hep: convert params from DD4hep unit to mm (mm is legacy expected by PPS reco software).
 */
DiamondDimensions DetGeomDesc::computeDiamondDimensions(const bool isABox,
                                                        const bool isDD4hep,
                                                        const std::vector<double>& params) const {
  DiamondDimensions boxShapeParameters{};
  if (isABox) {
    if (!isDD4hep) {
      // mm (old DD)
      boxShapeParameters = {params.at(0), params.at(1), params.at(2)};
    } else {
      // convert from DD4hep unit to mm (mm is legacy expected by PPS reco software)
      boxShapeParameters = {params.at(0) / dd4hep::mm, params.at(1) / dd4hep::mm, params.at(2) / dd4hep::mm};
    }
  }
  return boxShapeParameters;
}

/*
 * old DD DetId computation.
 * Relies on name and volumes copy numbers.
 */
DetId DetGeomDesc::computeDetID(const std::string& name,
                                const std::vector<int>& copyNos,
                                const unsigned int copyNum,
                                const bool isRun2) const {
  DetId geoID;

  // strip sensors
  if (name == DDD_TOTEM_RP_SENSOR_NAME) {
    // check size of copy numbers vector
    if (copyNos.size() < 3)
      throw cms::Exception("DDDTotemRPConstruction")
          << "size of copyNumbers for strip sensor is " << copyNos.size() << ". It must be >= 3.";

    // extract information
    const unsigned int decRPId = copyNos[copyNos.size() - 3];
    const unsigned int arm = decRPId / 100;
    const unsigned int station = (decRPId % 100) / 10;
    const unsigned int rp = decRPId % 10;
    const unsigned int detector = copyNos[copyNos.size() - 1];
    geoID = TotemRPDetId(arm, station, rp, detector);
  }

  // strip and pixels RPs
  else if (name == DDD_TOTEM_RP_RP_NAME || name == DDD_CTPPS_PIXELS_RP_NAME) {
    unsigned int decRPId = copyNum;

    // check if it is a pixel RP
    if (decRPId >= 10000) {
      decRPId = decRPId % 10000;
      const unsigned int armIdx = (decRPId / 100) % 10;
      const unsigned int stIdx = (decRPId / 10) % 10;
      const unsigned int rpIdx = decRPId % 10;
      geoID = CTPPSPixelDetId(armIdx, stIdx, rpIdx);
    } else {
      const unsigned int armIdx = (decRPId / 100) % 10;
      const unsigned int stIdx = (decRPId / 10) % 10;
      const unsigned int rpIdx = decRPId % 10;
      geoID = TotemRPDetId(armIdx, stIdx, rpIdx);
    }
  }

  else if (std::regex_match(name, std::regex(DDD_TOTEM_TIMING_SENSOR_TMPL))) {
    // check size of copy numbers vector
    if (copyNos.size() < 4)
      throw cms::Exception("DDDTotemRPConstruction")
          << "size of copyNumbers for TOTEM timing sensor is " << copyNos.size() << ". It must be >= 4.";

    const unsigned int decRPId = copyNos[copyNos.size() - 4];
    const unsigned int arm = decRPId / 100, station = (decRPId % 100) / 10, rp = decRPId % 10;
    const unsigned int plane = copyNos[copyNos.size() - 2], channel = copyNos[copyNos.size() - 1];
    geoID = TotemTimingDetId(arm, station, rp, plane, channel);
  }

  else if (name == DDD_TOTEM_TIMING_RP_NAME) {
    const unsigned int arm = copyNum / 100, station = (copyNum % 100) / 10, rp = copyNum % 10;
    geoID = TotemTimingDetId(arm, station, rp);
  }

  // pixel sensors
  else if (name == DDD_CTPPS_PIXELS_SENSOR_NAME || name == DDD_CTPPS_PIXELS_SENSOR_NAME_2x2) {
    // check size of copy numbers vector
    if (copyNos.size() < 4)
      throw cms::Exception("DDDTotemRPConstruction")
          << "size of copyNumbers for pixel sensor is " << copyNos.size() << ". It must be >= 4.";

    // extract information
    const unsigned int decRPId = copyNos[copyNos.size() - 4] % 10000;
    const unsigned int arm = decRPId / 100;
    const unsigned int station = (decRPId % 100) / 10;
    const unsigned int rp = decRPId % 10;
    const unsigned int detector = copyNos[copyNos.size() - 2] - 1;
    geoID = CTPPSPixelDetId(arm, station, rp, detector);
  }

  // diamond/UFSD sensors
  else if (name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME || name == DDD_CTPPS_UFSD_SEGMENT_NAME) {
    // check size of copy numbers vector
    if (copyNos.size() < 2)
      throw cms::Exception("DDDTotemRPConstruction")
          << "size of copyNumbers for diamond segments is " << copyNos.size() << ". It must be >= 2.";
    const unsigned int decRPId = copyNos[1];
    unsigned int arm, station, rp;
    if (isRun2) {
      arm = decRPId - 1;
      station = 1;
      rp = 6;
    } else {
      arm = (decRPId % 1000) / 100;
      station = (decRPId % 100) / 10;
      rp = decRPId % 10;
    }
    const unsigned int id = copyNos[copyNos.size() - 1];
    const unsigned int plane = id / 100;
    const unsigned int channel = id % 100;
    geoID = CTPPSDiamondDetId(arm, station, rp, plane, channel);
  }

  // diamond/UFSD RPs
  else if (name == DDD_CTPPS_DIAMONDS_RP_NAME) {
    // check size of copy numbers vector
    if (copyNos.size() < 2)
      throw cms::Exception("DDDTotemRPConstruction")
          << "size of copyNumbers for diamond RP is " << copyNos.size() << ". It must be >= 2.";

    const unsigned int decRPId = copyNos[1];
    unsigned int arm, station, rp;
    if (isRun2) {
      arm = decRPId - 1;
      station = 1;
      rp = 6;
    } else {
      arm = (decRPId % 1000) / 100;
      station = (decRPId % 100) / 10;
      rp = decRPId % 10;
    }
    geoID = CTPPSDiamondDetId(arm, station, rp);
  }

  return geoID;
}

/*
 * DD4hep DetId computation.
 */
DetId DetGeomDesc::computeDetIDFromDD4hep(const std::string& name,
                                          const std::vector<int>& copyNos,
                                          const unsigned int copyNum,
                                          const bool isRun2) const {
  std::vector<int> copyNosOldDD = {copyNos.rbegin() + 1, copyNos.rend()};

  return computeDetID(name, copyNosOldDD, copyNum, isRun2);
}

/*
 * Sensor type computation.
 * Find out from the namespace (from DB) or the volume name (from XMLs), whether a sensor type is 2x2.
 */
std::string DetGeomDesc::computeSensorType(std::string_view name) {
  std::string sensorType;

  // Case A: Construction from DB.
  // Namespace is present, and allow identification of 2x2 sensor type: just look for "2x2:RPixWafer" in name.
  const auto& foundFromDB = name.find(DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2 + ":" + DDD_CTPPS_PIXELS_SENSOR_NAME);
  if (foundFromDB != std::string::npos) {
    sensorType = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
  }

  // Case B: Construction from XMLs.
  // Volume name allows identification of 2x2 sensor type: just look whether name is "RPixWafer2x2".
  const auto& foundFromXML = name.find(DDD_CTPPS_PIXELS_SENSOR_NAME_2x2);
  if (foundFromXML != std::string::npos) {
    sensorType = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
  }

  return sensorType;
}
