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

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "TGeoMatrix.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"

//#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"

using namespace std;

using namespace cms_units::operators;

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc(DDFilteredView* fv)
  : m_trans(fv->translation()),
    m_rot(fv->rotation()),
    m_name(((fv->logicalPart()).ddname()).name()),
    m_params(((fv->logicalPart()).solid()).parameters()),
    m_copy(fv->copyno()),
    m_z(fv->geoHistory().back().absTranslation().z()),
    m_sensorType("") {
  std::string sensor_name = fv->geoHistory().back().logicalPart().name().fullname();
  std::size_t found = sensor_name.find(DDD_CTPPS_PIXELS_SENSOR_NAME);
  if (found != std::string::npos && sensor_name.substr(found - 4, 3) == DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2) {
    m_sensorType = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
  }
}

//----------------------------------------------------------------------------------------------------

// Constructor from DD4Hep DDFilteredView

DetGeomDesc::DetGeomDesc(cms::DDFilteredView* fv)
  /*: m_trans(fv->translation()),
    m_rot(fv->rotation()),
    m_name(fv->name()),
    //      m_params(((fv->logicalPart()).solid()).parameters()),
    m_copy(fv->copyNum()),
    m_z((fv->geoHistory().front()->GetMatrix()->GetTranslation())[2]),
    m_sensorType("") {
    //  std::string sensor_name = fv->geoHistory().back().logicalPart().name().fullname();
    /*
    std::string sensor_name = fv->history().back().logicalPart().name().fullname();
    std::size_t found = sensor_name.find(DDD_CTPPS_PIXELS_SENSOR_NAME);
    if (found != std::string::npos && sensor_name.substr(found - 4, 3) == DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2) {
    m_sensorType = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
    }
  */



  : m_trans(fv->translation() / 1._mm),  // Convert cm (DD4hep) to mm (legacy)
    m_rot(fv->rotation()),
    m_name(fv->name()),
    //m_params = ((fv->volume()).solid()).parameters(); TO DOOOOOOOOOOOOOOOOOOOOOOOOO
    //m_params(fv->parameters()),
    

    m_geographicalID(computeDetID(fv)),
    m_copy(fv->copyNum()),
    //m_z = fv->geoHistory().back().absTranslation().z();
    m_z(fv->translation().z() / 1._mm),  // Convert cm (DD4hep) to mm (legacy)
    m_sensorType("") {


  const cms::DDSolidShape& mySolidShape = cms::dd::getCurrentShape(*fv);
  std::cout << "fv->getMyName() = " << fv->getMyName() << std::endl;

  if (mySolidShape == cms::DDSolidShape::ddbox) {
    const cms::dd::DDBox& myShape = cms::dd::DDBox(*fv);
    m_params = { myShape.halfX() / 1._mm,
		 myShape.halfY() / 1._mm,
		 myShape.halfZ() / 1._mm
    }; 
  }

  else if (mySolidShape == cms::DDSolidShape::ddcons) {
    const cms::dd::DDCons& myShape = cms::dd::DDCons(*fv);
    m_params = { myShape.zhalf() / 1._mm,
		 myShape.rInMinusZ() / 1._mm,
		 myShape.rOutMinusZ() / 1._mm,
		 myShape.rInPlusZ() / 1._mm,
		 myShape.rOutPlusZ() / 1._mm,
		 myShape.phiFrom(),
		 myShape.deltaPhi()
    }; 
  }
  else if (mySolidShape == cms::DDSolidShape::ddtrap) {
    const cms::dd::DDTrap& myShape = cms::dd::DDTrap(*fv);
    m_params = { myShape.halfZ() / 1._mm,
		 myShape.theta(),
		 myShape.phi(),
		 myShape.y1() / 1._mm,
		 myShape.x1() / 1._mm,
		 myShape.x2() / 1._mm,
		 myShape.alpha1(),
		 myShape.y2() / 1._mm,
		 myShape.x3() / 1._mm,
		 myShape.x4() / 1._mm,		 
		 myShape.alpha2()
    }; 
  }
  else if (mySolidShape == cms::DDSolidShape::ddtubs) {
    const cms::dd::DDTubs& myShape = cms::dd::DDTubs(*fv);
    m_params = { myShape.zhalf() / 1._mm,
		 myShape.rIn() / 1._mm,
		 myShape.rOut() / 1._mm,
		 myShape.startPhi(),
		 myShape.deltaPhi()
    };
  }
  else if (mySolidShape == cms::DDSolidShape::ddtrunctubs) {
    const cms::dd::DDTruncTubs& myShape = cms::dd::DDTruncTubs(*fv);
    m_params = { myShape.zHalf() / 1._mm,
		 myShape.rIn() / 1._mm,
		 myShape.rOut() / 1._mm,
		 myShape.startPhi(),
		 myShape.deltaPhi(),
		 myShape.cutAtStart() / 1._mm,
		 myShape.cutAtDelta() / 1._mm,
		 static_cast<double>(myShape.cutInside())
    }; 
  }
  else if (mySolidShape == cms::DDSolidShape::dd_not_init) {
    std::cout << "DetGeomDesc::DetGeomDesc(cms::DDFilteredView* fv): ERROR: shape not supported for " 
	      << m_name << ", Id = " << m_geographicalID
	      << std::endl;
  }





  /*
    std::cout << "DetGeomDesc::DetGeomDesc m_name = " << m_name << std::endl;
    std::cout << "view->copyNumbers() = ";
    for (const auto& num : fv->copyNumbers()) {
    std::cout << num << " ";
    }
    std::cout << " " << std::endl;*/

 
  //const std::string sensor_name {fv->name()};
  //const std::string sensor_name = fv->fullname();
  //std::string sensor_name = fv->geoHistory().back().logicalPart().name().fullname();
  /* std::size_t found = sensor_name.find(DDD_CTPPS_PIXELS_SENSOR_NAME);
     if (found != std::string_view::npos && sensor_name.substr(found - 4, 3) == DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2) {
     m_sensorType = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
     }*/

}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::DetGeomDesc(PDetGeomDesc* pd) {
  for (auto i : pd->container_) {
    
    DetGeomDesc* gd = new DetGeomDesc();
    
    gd->setTranslation(i.dx_,i.dy_,i.dz_);
    gd->setRotation(i.axx_,i.axy_,i.axz_,
                    i.ayx_,i.ayy_,i.ayz_,
                    i.azx_,i.azy_,i.azz_);
    gd->setName(i.name_);
    gd->setParams(i.params_);
    gd->setGeographicalID(i.geographicalID_);
    gd->setCopyno(i.copy_);
    gd->setParentZPosition(i.z_);
    gd->setSensorType(i.sensorType_);
    
    this->addComponent(gd);
  }
}

//----------------------------------------------------------------------------------------------------
DetGeomDesc::DetGeomDesc(const DetGeomDesc& ref) { (*this) = ref; }

//----------------------------------------------------------------------------------------------------

DetGeomDesc& DetGeomDesc::operator=(const DetGeomDesc& ref) {
  m_params = ref.m_params;
  m_trans = ref.m_trans;
  m_rot = ref.m_rot;
  m_name = ref.m_name;
  m_copy = ref.m_copy;
  m_geographicalID = ref.m_geographicalID;
  m_z = ref.m_z;
  m_sensorType = ref.m_sensorType;
  return (*this);
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc::~DetGeomDesc() { deepDeleteComponents(); }

//----------------------------------------------------------------------------------------------------

DetGeomDesc::Container DetGeomDesc::components() const { return m_container; }

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::addComponent(DetGeomDesc* det) { m_container.emplace_back(det); }

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::deleteComponents() { m_container.erase(m_container.begin(), m_container.end()); }

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::deepDeleteComponents() {
  for (auto& it : m_container) {
    it->deepDeleteComponents();
    delete it;
  }
  clearComponents();
}

//----------------------------------------------------------------------------------------------------

void DetGeomDesc::applyAlignment(const CTPPSRPAlignmentCorrectionData& t) {
  m_rot = t.getRotationMatrix() * m_rot;
  m_trans = t.getTranslation() + m_trans;
}


DetId DetGeomDesc::computeDetID(cms::DDFilteredView* fv) {
  DetId geoID;


  std::string name(fv->name());

  // strip sensors
  if (name == DDD_TOTEM_RP_SENSOR_NAME) {
    const std::vector<int>& copy_num = fv->copyNos();
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
    unsigned int decRPId = fv->copyNum();

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
    const std::vector<int>& copy_num = fv->copyNos();
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
    const unsigned int arm = fv->copyNum() / 100, station = (fv->copyNum() % 100) / 10, rp = fv->copyNum() % 10;
    geoID = TotemTimingDetId(arm, station, rp);
  }

  // pixel sensors
  else if (name == DDD_CTPPS_PIXELS_SENSOR_NAME) {
    const std::vector<int>& copy_num = fv->copyNos();
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
    const std::vector<int>& copy_num = fv->copyNos();

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
    const std::vector<int>& copy_num = fv->copyNos();

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
