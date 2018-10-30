#include <typeinfo>

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iostream>
#include <map>

namespace {
  GeomDetEnumerators::SubDetector
  geometricDetToGeomDet(GeometricTimingDet::GTDEnumType gdenum) {
    // provide a map between the GeometricTimingDet enumerators and the GeomDet enumerators of the possible tracker subdetectors
    if(gdenum == GeometricTimingDet::GTDEnumType::BTL ) return GeomDetEnumerators::SubDetector::TimingBarrel;
    if(gdenum == GeometricTimingDet::GTDEnumType::ETL ) return GeomDetEnumerators::SubDetector::TimingEndcap;    
    return GeomDetEnumerators::SubDetector::invalidDet;
  }
  
  class DetIdComparator {
  public:
    bool operator()(GeometricTimingDet const* gd1, GeometricTimingDet const * gd2) const {
      uint32_t det1 = gd1->geographicalId();
      uint32_t det2 = gd2->geographicalId();
      return det1 < det2;
    }  
  };
}

MTDGeometry::MTDGeometry(GeometricTimingDet const* gd)
   : theTrackerDet(gd)
{
  for(unsigned int i=0;i<2;++i) {
    theSubDetTypeMap[i] = GeomDetEnumerators::invalidDet;
    theNumberOfLayers[i] = 0;
  }
  GeometricTimingDet::ConstGeometricTimingDetContainer subdetgd = gd->components();
  
  LogDebug("BuildingSubDetTypeMap") 
    << "GeometricTimingDet and GeomDetEnumerators enumerator values of the subdetectors" << std::endl;
  for(unsigned int i=0;i<subdetgd.size();++i) {
    MTDDetId mtdid(subdetgd[i]->geographicalId());
    assert(mtdid.mtdSubDetector()>0 && mtdid.mtdSubDetector()<3);
    theSubDetTypeMap[mtdid.mtdSubDetector()-1] = geometricDetToGeomDet(subdetgd[i]->type());
    theNumberOfLayers[mtdid.mtdSubDetector()-1]= subdetgd[i]->components().size();   
    LogTrace("BuildingSubDetTypeMap").log( [&](auto & debugstr) { 
	debugstr << "subdet " << i 
		 << " Geometric Det type " << subdetgd[i]->type()
		 << " Geom Det type " << theSubDetTypeMap[mtdid.mtdSubDetector()-1]
		 << " detid " << std::hex <<  subdetgd[i]->geographicalId().rawId() << std::dec
		 << " subdetid " <<  mtdid.mtdSubDetector()
		 << " number of layers " << subdetgd[i]->components().size()
		 << std::endl;
      });
  }
  LogDebug("SubDetTypeMapContent").log( [&](auto & debugstr) {
      debugstr << "Content of theSubDetTypeMap" << std::endl;
      for(unsigned int i=1;i<=2;++i) {	
	debugstr << " detid subdet "<< i 
		 << " Geom Det type "<< geomDetSubDetector(i) << std::endl; 
      }
    });
  LogDebug("NumberOfLayers").log( [&](auto & debugstr) { 
      debugstr << "Content of theNumberOfLayers" << std::endl;
      for(unsigned int i=1;i<=2;++i) {
	debugstr << " detid subdet "<< i 
		 << " number of layers " << numberOfLayers(i) << std::endl; 
      }
    });
  std::vector<const GeometricTimingDet*> deepcomp;
  gd->deepComponents(deepcomp);
   
  sort(deepcomp.begin(), deepcomp.end(), DetIdComparator());

  LogDebug("ThicknessAndType") 
    << " Total Number of Detectors " << deepcomp.size() << std::endl;
  LogDebug("ThicknessAndType") 
    << "Dump of sensors names and bounds" << std::endl;
  LogDebug("ThicknessAndType").log( [&](auto & debugstr) {
      for(auto det : deepcomp) {
	fillTestMap(det); 
	debugstr << std::hex << det->geographicalId().rawId() << std::dec
		 << " " << det->name().fullname() << " " 
		 << det->bounds()->thickness();
      }
    });
  LogDebug("DetTypeList").log( [&](auto & debugstr) {  
      debugstr << " Content of DetTypetList : size " << theDetTypetList.size() << std::endl;
      for (auto iVal : theDetTypetList) {
	debugstr 
	  << " DetId " << std::get<0>(iVal).rawId()
	  << " Type " << static_cast<std::underlying_type<MTDGeometry::ModuleType>::type>(std::get<1>(iVal))
	  << " Thickness " << std::get<2>(iVal) << std::endl;
      }  
    });
}

MTDGeometry::~MTDGeometry() {
    for (auto d : theDets) delete const_cast<GeomDet*>(d);
    for (auto d : theDetTypes) delete const_cast<GeomDetType*>(d);
}

void MTDGeometry::finalize() {
    theDetTypes.shrink_to_fit();  // owns the DetTypes
    theDetUnits.shrink_to_fit();  // they're all also into 'theDets', so we assume 'theDets' owns them
    theDets.shrink_to_fit();     // owns *ONLY* the GeomDet * corresponding to GluedDets.
    theDetUnitIds.shrink_to_fit();
    theDetIds.shrink_to_fit();
  
    theBTLDets.shrink_to_fit(); // not owned: they're also in 'theDets'
    theETLDets.shrink_to_fit(); // not owned: they're also in 'theDets'    
}

void MTDGeometry::addType(GeomDetType const * p) {
  theDetTypes.emplace_back(p);  // add to vector
}

void MTDGeometry::addDetUnit(GeomDet const * p) {
  // set index
  const_cast<GeomDet *>(p)->setIndex(theDetUnits.size()); 
  theDetUnits.emplace_back(p);  // add to vector  
  theMapUnit.emplace(p->geographicalId().rawId(),p);
}

void MTDGeometry::addDetUnitId(DetId p){
  theDetUnitIds.emplace_back(p);
}

void MTDGeometry::addDet(GeomDet const * p) {
  // set index
  const_cast<GeomDet *>(p)->setGdetIndex(theDets.size());
  theDets.emplace_back(p);  // add to vector
  theMap.insert(std::make_pair(p->geographicalId().rawId(),p));
  MTDDetId id(p->geographicalId());
  switch(id.mtdSubDetector()){
  case MTDDetId::BTL:
    theBTLDets.emplace_back(p);
    break;
  case MTDDetId::ETL:
    theETLDets.emplace_back(p);
    break;  
  default:
    edm::LogError("MTDGeometry")<<"ERROR - I was expecting a MTD Subdetector, I got a "<<id.mtdSubDetector();
  }
}

void MTDGeometry::addDetId(DetId p){
  theDetIds.emplace_back(p);
}


const MTDGeometry::DetContainer&
MTDGeometry::detsBTL() const
{
  return theBTLDets;
}

const MTDGeometry::DetContainer&
MTDGeometry::detsETL() const
{
  return theETLDets;
}

const MTDGeomDet * 
MTDGeometry::idToDetUnit(DetId s)const
{
  mapIdToDetUnit::const_iterator p=theMapUnit.find(s.rawId());
  if (p != theMapUnit.end()) {
    return static_cast<const MTDGeomDet *>(p->second);
  } 
  return nullptr;
}

const MTDGeomDet* 
MTDGeometry::idToDet(DetId s)const
{
  mapIdToDet::const_iterator p=theMap.find(s.rawId());
  if (p != theMap.end()) {
    return static_cast<const MTDGeomDet *>(p->second);
  }
  return nullptr;
}

const GeomDetEnumerators::SubDetector 
MTDGeometry::geomDetSubDetector(int subdet) const {
  if(subdet>=1 && subdet<=2) {
    return theSubDetTypeMap[subdet-1];
  } else {
    throw cms::Exception("WrongTrackerSubDet") << "Subdetector " << subdet;
  }
}

unsigned int
MTDGeometry::numberOfLayers(int subdet) const {
  if(subdet>=1 && subdet<=2) {
    return theNumberOfLayers[subdet-1];
  } else {
    throw cms::Exception("WrongTrackerSubDet") << "Subdetector " << subdet;
  }
}

bool
MTDGeometry::isThere(GeomDetEnumerators::SubDetector subdet) const {
  for(unsigned int i=1;i<=2;++i) {
    if(subdet == geomDetSubDetector(i)) return true;
  }
  return false;
}

void MTDGeometry::fillTestMap(const GeometricTimingDet* gd) {
    
  std::string temp = gd->name().fullname();
  std::string name = temp.substr(temp.find(":")+1); 
  DetId detid = gd->geographicalId();
  float thickness = gd->bounds()->thickness();
  std::string nameTag;  
  MTDGeometry::ModuleType mtype = moduleType(name);
  if (theDetTypetList.empty()) {
    theDetTypetList.emplace_back(detid, mtype, thickness);
  } else {
    auto  & t = (*(theDetTypetList.end()-1));
    if (std::get<1>(t) != mtype) theDetTypetList.emplace_back(detid, mtype, thickness);
    else {
      if  ( detid > std::get<0>(t) ) std::get<0>(t) = detid;
    }
  }
}

MTDGeometry::ModuleType MTDGeometry::getDetectorType(DetId detid) const {
  for (auto iVal : theDetTypetList) {
    DetId detid_max = std::get<0>(iVal);
    MTDGeometry::ModuleType mtype =  std::get<1>(iVal);     
    if (detid.rawId() <=  detid_max.rawId()) return mtype;
  }
  return MTDGeometry::ModuleType::UNKNOWN;
}

float MTDGeometry::getDetectorThickness(DetId detid) const {
  for (auto iVal : theDetTypetList) {
    DetId detid_max = std::get<0>(iVal);
    if (detid.rawId() <=  detid_max.rawId()) 
      return std::get<2>(iVal);
  }
  return -1.0;
}

MTDGeometry::ModuleType MTDGeometry::moduleType(const std::string& name) const {
  if ( name.find("Timing") != std::string::npos ){
    if ( name.find("BModule") != std::string::npos ) return ModuleType::BTL;
    else if ( name.find("EModule") != std::string::npos  ) return ModuleType::ETL;
  }
  return ModuleType::UNKNOWN;  
}
