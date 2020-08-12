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
#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"

#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "TGeoMatrix.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"


using namespace std;
using namespace cms_units::operators;


// Constructor from DD4Hep DDFilteredView, also using the SpecPars to access 2x2 wafers info.
//  A conversion factor (/1._mm) is applied wherever needed.
DetGeomDesc::DetGeomDesc(const cms::DDFilteredView& fv, const cms::DDSpecParRegistry& allSpecParSections)
  : m_name(fv.name()),
    m_copy(fv.copyNum()),
    m_trans(fv.translation() / 1._mm),  // Convert cm (DD4hep) to mm (legacy)
    m_rot(fv.rotation()),
    m_params(copyParameters(fv)),   
    m_isABox(fv.isABox()),
    m_sensorType(computeSensorType(fv.path(), allSpecParSections)),
    m_geographicalID(computeDetID(fv)),
    m_z(fv.translation().z() / 1._mm)  // Convert cm (DD4hep) to mm (legacy)
{}


DetGeomDesc::DetGeomDesc(const DetGeomDesc& ref) { (*this) = ref; }


DetGeomDesc& DetGeomDesc::operator=(const DetGeomDesc& ref) {
  m_name = ref.m_name;
  m_copy = ref.m_copy;
  m_trans = ref.m_trans;
  m_rot = ref.m_rot;
  m_params = ref.m_params;
  m_isABox = ref.m_isABox;
  m_sensorType = ref.m_sensorType;
  m_geographicalID = ref.m_geographicalID;
  m_z = ref.m_z;
  return (*this);
}


DetGeomDesc::~DetGeomDesc() { deepDeleteComponents(); }


DetGeomDesc::Container DetGeomDesc::components() const { return m_container; }


void DetGeomDesc::addComponent(DetGeomDesc* det) { m_container.emplace_back(det); }


DiamondDimensions DetGeomDesc::getDiamondDimensions() const {
  // Convert parameters units from cm (DD4hep standard) to mm (expected by PPS reco software).
  // This implementation is customized for the diamond sensors, which are represented by the 
  // Box shape parameterized by x, y and z half width.
  DiamondDimensions parameters;
  if (isABox()) {
    parameters = { m_params.at(0) / 1._mm, m_params.at(1) / 1._mm, m_params.at(2) / 1._mm };
  }
  else {
    edm::LogError("DetGeomDesc::getDiamondDimensions is not called on a box, for solid ")
      << name() << ", Id = " << geographicalID();
  }
  return parameters;
}


void DetGeomDesc::applyAlignment(const CTPPSRPAlignmentCorrectionData& t) {
  m_rot = t.getRotationMatrix() * m_rot;
  m_trans = t.getTranslation() + m_trans;
}


/*
 * private
 */


void DetGeomDesc::deleteComponents() { m_container.erase(m_container.begin(), m_container.end()); }


void DetGeomDesc::deepDeleteComponents() {
  for (auto& it : m_container) {
    it->deepDeleteComponents();
    delete it;
  }
  clearComponents();
}


std::vector<double> DetGeomDesc::copyParameters(const cms::DDFilteredView& fv) const {
  auto myShape = fv.solid();
  const std::vector<double>& parameters = myShape.dimensions();
  return parameters;
}


DetId DetGeomDesc::computeDetID(const cms::DDFilteredView& fv) const {
  DetId geoID;

  const std::string name{fv.name()};

  // strip sensors
  if (name == DDD_TOTEM_RP_SENSOR_NAME) {
    const std::vector<int>& copy_num = fv.copyNos();
    // check size of copy numbers array
    if (copy_num.size() < 4)
      throw cms::Exception("DDDTotemRPContruction")
	<< "size of copyNumbers for strip sensor is " << copy_num.size() << ". It must be >= 4.";

    // extract information
    const unsigned int decRPId = copy_num[2];
    const unsigned int arm = decRPId / 100;
    const unsigned int station = (decRPId % 100) / 10;
    const unsigned int rp = decRPId % 10;
    const unsigned int detector = copy_num[0];
    geoID = TotemRPDetId(arm, station, rp, detector);
  }

  // strip and pixels RPs
  else if (name == DDD_TOTEM_RP_RP_NAME || name == DDD_CTPPS_PIXELS_RP_NAME) {
    unsigned int decRPId = fv.copyNum();

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
    const std::vector<int>& copy_num = fv.copyNos();
    // check size of copy numbers array
    if (copy_num.size() < 5)
      throw cms::Exception("DDDTotemRPContruction")
	<< "size of copyNumbers for TOTEM timing sensor is " << copy_num.size() << ". It must be >= 5.";

    const unsigned int decRPId = copy_num[3];
    const unsigned int arm = decRPId / 100, station = (decRPId % 100) / 10, rp = decRPId % 10;
    const unsigned int plane = copy_num[1], channel = copy_num[0];
    geoID = TotemTimingDetId(arm, station, rp, plane, channel);
  }

  else if (name == DDD_TOTEM_TIMING_RP_NAME) {
    const unsigned int arm = fv.copyNum() / 100, station = (fv.copyNum() % 100) / 10, rp = fv.copyNum() % 10;
    geoID = TotemTimingDetId(arm, station, rp);
  }

  // pixel sensors
  else if (name == DDD_CTPPS_PIXELS_SENSOR_NAME) {
    const std::vector<int>& copy_num = fv.copyNos();
    // check size of copy numbers array
    if (copy_num.size() < 5)
      throw cms::Exception("DDDTotemRPContruction")
	<< "size of copyNumbers for pixel sensor is " << copy_num.size() << ". It must be >= 5.";

    // extract information
    const unsigned int decRPId = copy_num[3] % 10000;
    const unsigned int arm = decRPId / 100;
    const unsigned int station = (decRPId % 100) / 10;
    const unsigned int rp = decRPId % 10;
    const unsigned int detector = copy_num[1] - 1;
    geoID = CTPPSPixelDetId(arm, station, rp, detector);
  }

  // diamond/UFSD sensors
  else if (name == DDD_CTPPS_DIAMONDS_SEGMENT_NAME || name == DDD_CTPPS_UFSD_SEGMENT_NAME) {
    const std::vector<int>& copy_num = fv.copyNos();

    const unsigned int id = copy_num[0];
    const unsigned int arm = copy_num[copy_num.size()-3] - 1;
    const unsigned int station = 1;
    const unsigned int rp = 6;
    const unsigned int plane = (id / 100);
    const unsigned int channel = id % 100;

    geoID = CTPPSDiamondDetId(arm, station, rp, plane, channel);
  }

  // diamond/UFSD RPs
  else if (name == DDD_CTPPS_DIAMONDS_RP_NAME) {
    const std::vector<int>& copy_num = fv.copyNos();

    // check size of copy numbers array
    if (copy_num.size() < 3)
      throw cms::Exception("DDDTotemRPContruction")
	<< "size of copyNumbers for diamond RP is " << copy_num.size() << ". It must be >= 3.";

    const unsigned int arm = copy_num[(copy_num.size()-3)] - 1;
    const unsigned int station = 1;
    const unsigned int rp = 6;

    geoID = CTPPSDiamondDetId(arm, station, rp);
  }

  return geoID;
}


/*
 * If nodePath has a 2x2RPixWafer parameter defined in an XML SPecPar section, sensorType is 2x2.
 */
std::string DetGeomDesc::computeSensorType(const std::string& nodePath, const cms::DDSpecParRegistry& allSpecParSections) {
  std::string sensorType;

  const std::string& parameterName = DDD_CTPPS_2x2_RPIXWAFER_PARAMETER_NAME;

  cms::DDSpecParRefs filteredSpecParSections;
  allSpecParSections.filter(filteredSpecParSections, parameterName);
  for (const auto& mySpecParSection : filteredSpecParSections) {
    if (mySpecParSection->hasPath(nodePath)) {
      sensorType = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
    }
  }

  return sensorType;
}
