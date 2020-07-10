/****************************************************************************
 *
 * Author:
 *
 *  Wagner Carvalho (wcarvalh@cern.ch)
 *
 *  DB builder for PPS geometry
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDTranslation.h"
#include "DetectorDescription/DDCMS/interface/DDRotationMatrix.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardIdealGeometryRecord.h"

#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"

#include <regex>


using namespace cms_units::operators;

//----------------------------------------------------------------------------------------------------

class PPSGeometryBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit PPSGeometryBuilder(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void buildPDetGeomDesc(cms::DDFilteredView*, PDetGeomDesc*);
  uint32_t getGeographicalID(cms::DDFilteredView*);

  std::string compactViewTag_;
  edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
  edm::Service<cond::service::PoolDBOutputService> dbservice_;
};

//----------------------------------------------------------------------------------------------------

PPSGeometryBuilder::PPSGeometryBuilder(const edm::ParameterSet& iConfig)
    : compactViewTag_(iConfig.getUntrackedParameter<std::string>("compactViewTag", "XMLIdealGeometryESSource_CTPPS")) {}

//----------------------------------------------------------------------------------------------------

void PPSGeometryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  //edm::ESTransientHandle<cms::DDCompactView> cpv;
  edm::ESHandle<cms::DDCompactView> cpv;


  //iSetup.get<IdealGeometryRecord>().get(cpv);
  //Get DDCompactView from IdealGeometryRecord
  if (watcherIdealGeometry_.check(iSetup)) {
    std::cout << "Got IdealGeometryRecord" << std::endl;
    iSetup.get<IdealGeometryRecord>().get(compactViewTag_.c_str(), cpv);
  }

  
  const cms::DDDetector* mySystem = cpv->detector();
  if (mySystem) {
    std::cout << "mySystem->detectors().size() = " << mySystem->detectors().size() << std::endl;
    for (const auto& sub : mySystem->detectors()) {
      std::cout << "sub name = " << sub.first << std::endl;
    }
    
  }


  // Create DDFilteredView and apply the filter
  //DDPassAllFilter filter;
  //cms::DDFilteredView fv((*cpv), filter);

  const dd4hep::Volume& worldVolume = mySystem->worldVolume();
  cms::DDFilteredView fv(mySystem, worldVolume);
  //cms::DDFilter filter;
  //cms::DDFilteredView fv(*cpv, filter);



  //auto const& det = iRecord.get(detectorToken_);
  //cms::DDCompactView cpvf(*mySystem);
  // create DDFilteredView and apply the filter
  //cms::DDFilter filter;
  //cms::DDFilteredView fv(cpvf, filter);

  if (fv.next(0) == false) {
    edm::LogError("PPSGeometryBuilder") << "Filtered view is empty. Cannot build.";
  }


  // Persistent geometry data
  PDetGeomDesc* pdet = new PDetGeomDesc;
  // Build geometry
  buildPDetGeomDesc(&fv, pdet);

  // Save geometry in the database
  if (pdet->container_.empty()) {
    throw cms::Exception("PPSGeometryBuilder") << "PDetGeomDesc is empty, no geometry to save in the database.";
  } else {
    if (dbservice_.isAvailable()) {
      dbservice_->writeOne(pdet, dbservice_->beginOfTime(), "VeryForwardIdealGeometryRecord");
    } else {
      throw cms::Exception("PoolDBService required.");
    }
  }

  return;
}

//----------------------------------------------------------------------------------------------------//----------------------------------------------------------------------------------------------------

void PPSGeometryBuilder::buildPDetGeomDesc(cms::DDFilteredView* fv, PDetGeomDesc* gd) {
  // try to dive into next level
  //if (!fv->firstChild())
  //return;

  // loop over siblings in the level
  do {
    // Create a PDetGeomDesc::Item node and add it to the parent's (gd) list
    PDetGeomDesc::Item item;

    // =======================================================================
    // Fill Item
    // =======================================================================

    item.dx_ = fv->translation().X() / 1._mm; // Convert cm (DD4hep) to mm (legacy)
    item.dy_ = fv->translation().Y() / 1._mm; // Convert cm (DD4hep) to mm (legacy)
    item.dz_ = fv->translation().Z() / 1._mm; // Convert cm (DD4hep) to mm (legacy)
    const DDRotationMatrix& rot = fv->rotation();
    double xx, xy, xz, yx, yy, yz, zx, zy, zz;
    rot.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
    item.axx_ = xx;
    item.axy_ = xy;
    item.axz_ = xz;
    item.ayx_ = yx;
    item.ayy_ = yy;
    item.ayz_ = yz;
    item.azx_ = zx;
    item.azy_ = zy;
    item.azz_ = zz;
    item.name_ = fv->name();
    //item.params_ = ((fv->volume()).solid()).parameters(); TO DOOOOOOOOOOOOOOOOOOOOOOOOO
    //item.params_ = fv->parameters();
    item.copy_ = fv->copyNum();
    //item.z_ = fv->geoHistory().back().absTranslation().z();
    item.z_ = fv->translation().z() / 1._mm; // Convert cm (DD4hep) to mm (legacy)
    // Sensor Type
    item.sensorType_ = "";
    //std::string sensor_name = fv->geoHistory().back().logicalPart().name().fullname();
    const std::string sensor_name {fv->name()};
    //const std::string sensor_name = fv->fullname();
    /* std::size_t found = sensor_name.find(DDD_CTPPS_PIXELS_SENSOR_NAME);
    if (found != std::string_view::npos && sensor_name.substr(found - 4, 3) == DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2) {
      item.sensorType_ = DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2;
    }*/
    // Geographical ID
    item.geographicalID_ = getGeographicalID(fv);

    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
    std::cout << " " << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!    item.name_ = " << item.name_ << std::endl;
    std::cout << "item.copy_ = " << item.copy_ << std::endl;
    std::cout << "item.geographicalID_ = " << item.geographicalID_ << std::endl;
    std::cout << "item.z_ = " << item.z_ << std::endl;
    std::cout << "sensor_name = " << sensor_name << std::endl;
    std::cout << "item.sensorType_ = " << item.sensorType_ << std::endl;
    std::cout << "item.dx_ = " << item.dx_ << std::endl;
    std::cout << "item.dy_ = " << item.dy_ << std::endl;
    std::cout << "item.dz_ = " << item.dz_ << std::endl;
    std::cout << "rot = " << rot << std::endl;
    std::cout << "item.params_ = ";
    for (const auto& val : item.params_) {
      std::cout << val << " ";
    }
    std::cout << " " << std::endl;

    // =======================================================================

    // add component
    gd->container_.push_back(item);

    // recursion
    //buildPDetGeomDesc(fv, gd);
  } while (fv->next(0)); //while (fv->nextSibling()); 

  // go a level up
  //fv->parent();
}

//----------------------------------------------------------------------------------------------------//----------------------------------------------------------------------------------------------------

uint32_t PPSGeometryBuilder::getGeographicalID(cms::DDFilteredView* view) {
  uint32_t geoID = 0;
  const std::string name {view->name()};
  std::cout << "PPSGeometryBuilder::getGeographicalID name = " << name << std::endl;
  std::cout << "view->copyNum() = " << view->copyNum() << std::endl;
  std::cout << "view->copyNumbers() = ";
  for (const auto& num : view->copyNumbers()) {
    std::cout << num << " ";
  }
  std::cout << " " << std::endl;

  // strip sensors
  if (name == DDD_TOTEM_RP_SENSOR_NAME) {
    const std::vector<int>& copy_num = view->copyNumbers();
    // check size of copy numubers array
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
    unsigned int decRPId = view->copyNum();

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
    const std::vector<int>& copy_num = view->copyNumbers();
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
    const unsigned int arm = view->copyNum() / 100, station = (view->copyNum() % 100) / 10, rp = view->copyNum() % 10;
    geoID = TotemTimingDetId(arm, station, rp);
  }

  // pixel sensors
  else if (name == DDD_CTPPS_PIXELS_SENSOR_NAME) {
    const std::vector<int>& copy_num = view->copyNumbers();
    // check size of copy numubers array
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
    const std::vector<int>& copy_num = view->copyNumbers();

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
    const std::vector<int>& copy_num = view->copyNumbers();

    // check size of copy numubers array
    if (copy_num.size() < 3)
      throw cms::Exception("DDDTotemRPContruction")
          << "size of copyNumbers for diamond RP is " << copy_num.size() << ". It must be >= 3.";

    const unsigned int arm = copy_num[copy_num.size()-3] - 1;
    const unsigned int station = 1;
    const unsigned int rp = 6;

    geoID = CTPPSDiamondDetId(arm, station, rp);
  }

  return geoID;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPSGeometryBuilder);
