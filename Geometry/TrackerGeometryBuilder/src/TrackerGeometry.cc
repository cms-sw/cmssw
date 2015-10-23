

#include <typeinfo>

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iostream>
#include <map>

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

namespace {
  GeomDetEnumerators::SubDetector 
  geometricDetToGeomDet(GeometricDet::GDEnumType gdenum) {
    // provide a map between the GeometricDet enumerators and the GeomDet enumerators of the possible tracker subdetectors
    if(gdenum == GeometricDet::GDEnumType::PixelBarrel ) return GeomDetEnumerators::SubDetector::PixelBarrel;
    if(gdenum == GeometricDet::GDEnumType::PixelEndCap) return GeomDetEnumerators::SubDetector::PixelEndcap;
    if(gdenum == GeometricDet::GDEnumType::TIB) return GeomDetEnumerators::SubDetector::TIB;
    if(gdenum == GeometricDet::GDEnumType::TID) return GeomDetEnumerators::SubDetector::TID;
    if(gdenum == GeometricDet::GDEnumType::TOB) return GeomDetEnumerators::SubDetector::TOB;
    if(gdenum == GeometricDet::GDEnumType::TEC) return GeomDetEnumerators::SubDetector::TEC;
    if(gdenum == GeometricDet::GDEnumType::PixelPhase1Barrel) return GeomDetEnumerators::SubDetector::P1PXB;
    if(gdenum == GeometricDet::GDEnumType::PixelPhase1EndCap) return GeomDetEnumerators::SubDetector::P1PXEC;
    if(gdenum == GeometricDet::GDEnumType::PixelPhase2EndCap) return GeomDetEnumerators::SubDetector::P2PXEC;
    if(gdenum == GeometricDet::GDEnumType::OTPhase2Barrel) return GeomDetEnumerators::SubDetector::P2OTB;
    if(gdenum == GeometricDet::GDEnumType::OTPhase2EndCap) return GeomDetEnumerators::SubDetector::P2OTEC;
    return GeomDetEnumerators::SubDetector::invalidDet;
  }
  class DetIdComparator {
  public:
    bool operator()(GeometricDet const* gd1, GeometricDet const * gd2) const {
      uint32_t det1 = gd1->geographicalId();
      uint32_t det2 = gd2->geographicalId();
      return det1 < det2;
    }
  };
  
}

TrackerGeometry::TrackerGeometry(GeometricDet const* gd) :  theTrackerDet(gd)
{
  for(unsigned int i=0;i<6;++i) {
    theSubDetTypeMap[i] = GeomDetEnumerators::invalidDet;
    theNumberOfLayers[i] = 0;
  }
  GeometricDet::ConstGeometricDetContainer subdetgd = gd->components();
  LogDebug("BuildingSubDetTypeMap") << "GeometriDet and GeomDetEnumerators enumerator values of the subdetectors";
  for(unsigned int i=0;i<subdetgd.size();++i) {
    assert(subdetgd[i]->geographicalId().subdetId()>0 && subdetgd[i]->geographicalId().subdetId()<7);
    theSubDetTypeMap[subdetgd[i]->geographicalId().subdetId()-1]= geometricDetToGeomDet(subdetgd[i]->type());
    theNumberOfLayers[subdetgd[i]->geographicalId().subdetId()-1]= subdetgd[i]->components().size();
    LogTrace("BuildingSubDetTypeMap") << "subdet " << i 
				      << " Geometric Det type " << subdetgd[i]->type()
				      << " Geom Det type " << theSubDetTypeMap[subdetgd[i]->geographicalId().subdetId()-1]
				      << " detid " <<  subdetgd[i]->geographicalId()
				      << " subdetid " <<  subdetgd[i]->geographicalId().subdetId()
				      << " number of layers " << subdetgd[i]->components().size();
  }
  LogDebug("SubDetTypeMapContent") << "Content of theSubDetTypeMap";
  for(unsigned int i=1;i<7;++i) {
    LogTrace("SubDetTypeMapContent") << " detid subdet "<< i << " Geom Det type " << geomDetSubDetector(i); 
  }
  LogDebug("NumberOfLayers") << "Content of theNumberOfLayers";
  for(unsigned int i=1;i<7;++i) {
    LogTrace("NumberOfLayers") << " detid subdet "<< i << " number of layers " << numberOfLayers(i); 
  }
  // checking GeometricDet tree leaves name and bounds
  
  std::vector<const GeometricDet*> deepcomp;
  gd->deepComponents(deepcomp);
   
  sort(deepcomp.begin(), deepcomp.end(), DetIdComparator());

  std::cout << " Total Number of Detectors " << deepcomp.size() << std::endl;  
  LogDebug("ThicknessAndType") << "Dump of sensors names and bounds";
  for(auto det : deepcomp) {
    fillTestMap(det); 
    LogDebug("ThicknessAndType") << det->geographicalId() << " " << det->name().fullname() << " " << det->bounds()->thickness();
  }
  LogDebug("DetTypeList") << " Content of DetTypetList : size " << theDetTypetList.size();
  for (auto iVal : theDetTypetList) {
    LogDebug("DetTypeList") << " DetId " <<  std::get<0>(iVal) << " Type " << std::get<1>(iVal)<< " Thickness " << std::get<2>(iVal);
  }
  
}

TrackerGeometry::~TrackerGeometry() {
    for (DetContainer::iterator     it = theDets.begin(),     ed = theDets.end();     it != ed; ++it) delete *it;
    for (DetTypeContainer::iterator it = theDetTypes.begin(), ed = theDetTypes.end(); it != ed; ++it) delete *it;
}

GeometricDet const * TrackerGeometry::trackerDet() const {
  return  theTrackerDet;
}


void TrackerGeometry::addType(GeomDetType* p) {
  theDetTypes.push_back(p);  // add to vector
}

void TrackerGeometry::addDetUnit(GeomDetUnit* p) {
  // set index
  p->setIndex(theDetUnits.size());
  theDetUnits.push_back(p);  // add to vector
  theMapUnit.insert(std::make_pair(p->geographicalId().rawId(),p));
}

void TrackerGeometry::addDetUnitId(DetId p){
  theDetUnitIds.push_back(p);
}

void TrackerGeometry::addDet(GeomDet* p) {
  theDets.push_back(p);  // add to vector
  theMap.insert(std::make_pair(p->geographicalId().rawId(),p));
  DetId id(p->geographicalId());
  switch(id.subdetId()){
  case PixelSubdetector::PixelBarrel:
    thePXBDets.push_back(p);
    break;
  case PixelSubdetector::PixelEndcap:
    thePXFDets.push_back(p);
    break;
  case StripSubdetector::TIB:
    theTIBDets.push_back(p);
    break;
  case StripSubdetector::TID:
    theTIDDets.push_back(p);
    break;
  case StripSubdetector::TOB:
    theTOBDets.push_back(p);
    break;
  case StripSubdetector::TEC:
    theTECDets.push_back(p);
    break;
  default:
    edm::LogError("TrackerGeometry")<<"ERROR - I was expecting a Tracker Subdetector, I got a "<<id.subdetId();
  }


}

void TrackerGeometry::addDetId(DetId p){
  theDetIds.push_back(p);
}

const TrackerGeometry::DetUnitContainer&
TrackerGeometry::detUnits() const
{
  return theDetUnits;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::dets() const
{
  return theDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsPXB() const
{
  return thePXBDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsPXF() const
{
  return thePXFDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTIB() const
{
  return theTIBDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTID() const
{
  return theTIDDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTOB() const
{
  return theTOBDets;
}

const TrackerGeometry::DetContainer&
TrackerGeometry::detsTEC() const
{
  return theTECDets;
}

const GeomDetUnit* 
TrackerGeometry::idToDetUnit(DetId s)const
{
  mapIdToDetUnit::const_iterator p=theMapUnit.find(s.rawId());
  if (p != theMapUnit.end())
    return (p)->second;
  edm::LogError("TrackerGeometry")<<"Invalid DetID: no GeomDetUnit associated";
  GeomDetUnit* geom = 0;
  return geom;
}

const GeomDet* 
TrackerGeometry::idToDet(DetId s)const
{
  mapIdToDet::const_iterator p=theMap.find(s.rawId());
  if (p != theMap.end())
    return (p)->second;
  edm::LogError("TrackerGeometry")<<"Invalid DetID: no GeomDet associated";
  GeomDet* geom = 0;
  return geom;
}

const GeomDetEnumerators::SubDetector 
TrackerGeometry::geomDetSubDetector(int subdet) const {
  if(subdet>=1 && subdet<=6) {
    return theSubDetTypeMap[subdet-1];
  } else {
    throw cms::Exception("WrongTrackerSubDet") << "Subdetector " << subdet;
  }
}

unsigned int
TrackerGeometry::numberOfLayers(int subdet) const {
  if(subdet>=1 && subdet<=6) {
    return theNumberOfLayers[subdet-1];
  } else {
    throw cms::Exception("WrongTrackerSubDet") << "Subdetector " << subdet;
  }
}

bool
TrackerGeometry::isThere(GeomDetEnumerators::SubDetector subdet) const {
  for(unsigned int i=1;i<7;++i) {
    if(subdet == geomDetSubDetector(i)) return true;
  }
  return false;
}

const TrackerGeometry::DetTypeContainer&  
TrackerGeometry::detTypes()   const 
{
  return theDetTypes;
}


const TrackerGeometry::DetIdContainer&  
TrackerGeometry::detUnitIds()   const 
{
  return theDetUnitIds;
}

const TrackerGeometry::DetIdContainer&  
TrackerGeometry::detIds()   const 
{
  return theDetIds;
}
void TrackerGeometry::fillTestMap(const GeometricDet* gd) {
    
  std::string temp = gd->name();
  std::string name = temp.substr(temp.find(":")+1); 
  DetId detid = gd->geographicalId();
  float thickness = gd->bounds()->thickness();
  std::string nameTag;  
  TrackerGeometry::ModuleType mtype = moduleType(name);
  if (theDetTypetList.size() == 0) {
    theDetTypetList.push_back({std::make_tuple(detid, mtype, thickness)});
  } else {
    auto  & t = (*(theDetTypetList.end()-1));
    if (std::get<1>(t) != mtype) theDetTypetList.push_back({std::make_tuple(detid, mtype, thickness)});
    else {
      if  ( detid > std::get<0>(t) ) std::get<0>(t) = detid;
    }
  }
}

TrackerGeometry::ModuleType TrackerGeometry::getDetectorType(DetId detid) const {
  for (auto iVal : theDetTypetList) {
    DetId detid_max = std::get<0>(iVal);
    TrackerGeometry::ModuleType mtype =  std::get<1>(iVal);     
    if (detid.rawId() <=  detid_max.rawId()) return mtype;
  }
  return TrackerGeometry::ModuleType::UNKNOWN;
}
float TrackerGeometry::getDetectorThickness(DetId detid) const {
  for (auto iVal : theDetTypetList) {
    DetId detid_max = std::get<0>(iVal);
    if (detid.rawId() <=  detid_max.rawId()) 
      return std::get<2>(iVal);
  }
  return -1.0;
}

TrackerGeometry::ModuleType TrackerGeometry::moduleType(std::string name) const {
  if ( name.find("PixelBarrel") != std::string::npos) return ModuleType::Ph1PXB;
  else if (name.find("PixelForward") != std::string::npos) return ModuleType::Ph1PXF;
  else if ( name.find("TIB") != std::string::npos) {
    if ( name.find("0") != std::string::npos) return ModuleType::IB1;
    else return ModuleType::IB2;
  } else if ( name.find("TOB") != std::string::npos) {
    if ( name.find("0") != std::string::npos) return ModuleType::OB1;
    else return ModuleType::OB2;
  } else if ( name.find("TID") != std::string::npos) {
    if ( name.find("0") != std::string::npos) return ModuleType::W1A; 
    else if ( name.find("1") != std::string::npos) return ModuleType::W2A;
    else if ( name.find("2") != std::string::npos) return ModuleType::W3A;
  } else if ( name.find("TEC") != std::string::npos) { 
    if ( name.find("0") != std::string::npos) return ModuleType::W1B;
    else if ( name.find("1") != std::string::npos) return ModuleType::W2B;
    else if ( name.find("2") != std::string::npos) return ModuleType::W3B;
    else if ( name.find("3") != std::string::npos) return ModuleType::W4;
    else if ( name.find("4") != std::string::npos) return ModuleType::W5;
    else if ( name.find("5") != std::string::npos) return ModuleType::W6;
    else if ( name.find("6") != std::string::npos) return ModuleType::W7;
  } else if ( name.find("BModule") != std::string::npos || name.find("EModule") != std::string::npos ) { 
    if (name.find("PSMacroPixel")) return ModuleType::Ph2PSP;
    else if (name.find("PSStrip")) return ModuleType::Ph2PSS;
    else if (name.find("2S")) return ModuleType::Ph2SS;
  }
  return ModuleType::UNKNOWN;  
}
