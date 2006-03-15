/// clhep
#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/Core/interface/DDExpandedView.h"
//temporary
#include "DetectorDescription/Core/interface/DDSolid.h"
//
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GeomTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderForGluedDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cfloat>
#include <vector>

using std::vector;
using std::string;


TrackerGeometry* TrackerGeomBuilderFromGeometricDet::build(const DDCompactView* cpv, const GeometricDet* gd){

  DDExpandedView ev(*cpv);

  TrackerGeometry* tracker = new TrackerGeometry();
  std::vector<const GeometricDet*> comp = gd->deepComponents();

  std::vector<const GeometricDet*> pixB; 
  std::vector<const GeometricDet*> pixF; 
  std::vector<const GeometricDet*> tib;  
  std::vector<const GeometricDet*> tid;  
  std::vector<const GeometricDet*> tob;  
  std::vector<const GeometricDet*> tec;  

  for(u_int32_t i = 0;i<comp.size();i++){
    u_int32_t answer = comp[i]->geographicalID().subdetId();
    if (answer == 1){
      pixB.push_back(comp[i]);
    }else if (answer == 2){
      pixF.push_back(comp[i]);
    }else if (answer == 3){
      tib.push_back(comp[i]);
    }else if (answer == 4){
      tid.push_back(comp[i]);
    }else if (answer == 5){
      tob.push_back(comp[i]);
    }else if (answer == 6){
      tec.push_back(comp[i]);
    }
    
  }
  buildPixelBarrel(pixB,&ev,tracker,theDetIdToEnum.type(1)); //"PixelBarrel" 
  buildPixelForward(pixF,&ev,tracker,theDetIdToEnum.type(2)); //"PixelEndcap" 
  buildSiliconBarrel(tib,&ev,tracker,theDetIdToEnum.type(3));// "TIB"	
  buildSiliconForward(tid,&ev,tracker,theDetIdToEnum.type(4));//"TID" 
  buildSiliconBarrel(tob,&ev,tracker,theDetIdToEnum.type(5));//"TOB"	
  buildSiliconForward(tec,&ev,tracker,theDetIdToEnum.type(6));//"TEC"        
  buildGeomDet(tracker);//"GeomDet"
  return tracker;
}

void TrackerGeomBuilderFromGeometricDet::buildPixelBarrel(std::vector<const GeometricDet*> gdv, DDExpandedView* evtemp,TrackerGeometry* tracker,GeomDetType::SubDetector& det){

  PlaneBuilderFromGeometricDet planeBuilder;
  static std::map<string,PixelGeomDetType*> detTypeMap;
  PixelGeomDetType* detType;
  DDExpandedView* ev = evtemp;

  for(u_int32_t i= 0;i<gdv.size();i++){
    ev->goTo(gdv[i]->navType());
    std::string detName = gdv[i]->name();

    if (detTypeMap.find(detName) == detTypeMap.end()) {
      PixelTopology* t;
      t =  theTopologyBuilder->buildPixel(gdv[i]->bounds(),
					  getDouble("PixelROCRows",ev),
					  getDouble("PixelROCCols",ev),
					  getDouble("PixelROC_X"  ,ev),
					  getDouble("PixelROC_Y"  ,ev),
					  "barrel");

      detType =  new PixelGeomDetType(t,detName,det);
      detTypeMap[detName]  =  detType;
      tracker->addType(detType);
    }
    
    PlaneBuilderFromGeometricDet::ResultType plane = planeBuilder.plane(gdv[i]);
    GeomDetUnit* temp = new PixelGeomDetUnit(&(*plane),detTypeMap[detName],gdv[i]);

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());

  }
  
}
    

void TrackerGeomBuilderFromGeometricDet::buildPixelForward(std::vector<const GeometricDet*> gdv, DDExpandedView* evtemp,TrackerGeometry* tracker,GeomDetType::SubDetector& det){ 

  PlaneBuilderFromGeometricDet planeBuilder;
  static std::map<std::string,PixelGeomDetType*> detTypeMap;
  DDExpandedView* ev = evtemp;  

  for(u_int32_t i=0;i<gdv.size();i++){

    ev->goTo(gdv[i]->navType());
    std::string detName = gdv[i]->name();
    if (detTypeMap.find(detName) == detTypeMap.end()) {

      PixelTopology* t;
      t = theTopologyBuilder->buildPixel(gdv[i]->bounds(),
					 getDouble("PixelROCRows",ev),
					 getDouble("PixelROCCols",ev),
					 getDouble("PixelROC_X"  ,ev),
					 getDouble("PixelROC_Y"  ,ev),
					 "endcap");
      
      detTypeMap[detName]  = new  PixelGeomDetType(t,detName,det);
      tracker->addType(detTypeMap[detName]);
    
    }

    PlaneBuilderFromGeometricDet::ResultType plane = planeBuilder.plane(gdv[i]);
    GeomDetUnit* temp =  new PixelGeomDetUnit(&(*plane),detTypeMap[detName],gdv[i]);

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());


  }
}

void TrackerGeomBuilderFromGeometricDet::buildSiliconBarrel(std::vector<const GeometricDet*> gdv, DDExpandedView* evtemp,TrackerGeometry* tracker,GeomDetType::SubDetector& det){

  PlaneBuilderFromGeometricDet planeBuilder;
  static std::map<std::string,StripGeomDetType*> detTypeMap;
  DDExpandedView* ev = evtemp;

  for(u_int32_t i=0;i<gdv.size();i++){

    ev->goTo(gdv[i]->navType());
    std::string detName = gdv[i]->name();
    if (detTypeMap.find(detName) == detTypeMap.end()) {

      bool stereo = false;
      if(getString("TrackerStereoDetectors",ev)=="true"){
	stereo = true;
      }

      StripTopology* t;
      t = theTopologyBuilder->buildStrip(gdv[i]->bounds(),
					 getDouble("SiliconAPVNumber",ev),
					 "barrel");

      detTypeMap[detName] = new  StripGeomDetType( t,detName,det,
						   stereo);
      tracker->addType(detTypeMap[detName]);
    }

    PlaneBuilderFromGeometricDet::ResultType plane = planeBuilder.plane(gdv[i]);  
    GeomDetUnit* temp = new StripGeomDetUnit(&(*plane), detTypeMap[detName],gdv[i]);

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());
    
    

  }
}

void TrackerGeomBuilderFromGeometricDet::buildSiliconForward(std::vector<const GeometricDet*> gdv, DDExpandedView* evtemp,TrackerGeometry* tracker,GeomDetType::SubDetector& det){ 

  PlaneBuilderFromGeometricDet planeBuilder;
  static std::map<std::string,StripGeomDetType*> detTypeMap;
  DDExpandedView* ev = evtemp;
  
  for(u_int32_t i=0;i<gdv.size();i++){

    ev->goTo(gdv[i]->navType());
    std::string detName = gdv[i]->name();
    if (detTypeMap.find(detName) == detTypeMap.end()) {

      bool stereo = false;
      if(getString("TrackerStereoDetectors",ev)=="true"){
	stereo = true;
      }

      StripTopology* t;
      t =  theTopologyBuilder->buildStrip(gdv[i]->bounds(),
					  getDouble("SiliconAPVNumber",ev),
					  "endcap");


      detTypeMap[detName] = new  StripGeomDetType( t,detName,det,
						   stereo);
      tracker->addType(detTypeMap[detName]);
    }

    PlaneBuilderFromGeometricDet::ResultType plane = planeBuilder.plane(gdv[i]);  
    GeomDetUnit* temp = new StripGeomDetUnit(&(*plane), detTypeMap[detName],gdv[i]);

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());
  }  
}

void TrackerGeomBuilderFromGeometricDet::buildGeomDet(TrackerGeometry* tracker){
  PlaneBuilderForGluedDet gluedplaneBuilder;
  std::vector<GeomDetUnit*> gdu= tracker->detUnits();
  std::vector<DetId> gduId = tracker->detUnitIds();
  for(u_int32_t i=0;i<gdu.size();i++){
    unsigned int subdet_id = gduId[i].subdetId();
    StripSubdetector sidet( gduId[i].rawId());
    tracker->addDet((GeomDet*) gdu[i]);
    tracker->addDetId(gduId[i]);      
    if(sidet.glued()!=0&&sidet.stereo()==1){
      int partner_pos=-1;
      for(u_int32_t jj=0;jj<gduId.size();jj++){
	if(DetId(sidet.partnerDetId())== gduId[jj]) partner_pos=jj; 
      }
      const GeomDetUnit* dus = gdu[i];
      if(partner_pos==-1){
	edm::LogError("TrackerGeomBuilderFromGeometricDet") <<"No partner detector found";
	abort();
      }
      const GeomDetUnit* dum = gdu[partner_pos];
      std::vector<const GeomDetUnit *> glued;
      glued.push_back(dum);
      glued.push_back(dus);
      std::string part = "barrel";
      if(subdet_id==StripSubdetector::TEC||subdet_id==StripSubdetector::TID) part = "endcap";
      PlaneBuilderForGluedDet::ResultType plane = gluedplaneBuilder.plane(glued,part);
      GluedGeomDet* gluedDet = new GluedGeomDet(&(*plane),dum,dus);
      tracker->addDet((GeomDet*) gluedDet);
      tracker->addDetId(DetId(sidet.glued()));
      glued.clear();
    }
  }
}


std::string TrackerGeomBuilderFromGeometricDet::getString(const std::string s, DDExpandedView* ev){
    vector<std::string> temp;
    DDValue val(s);
    vector<const DDsvalues_type *> result = ev->specifics();
    vector<const DDsvalues_type *>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it)
    {
	foundIt = DDfetch(*it,val);
	if (foundIt) break;

    }    
    if (foundIt)
    { 
	temp = val.strings(); 
	if (temp.size() != 1)
	{
	  edm::LogError("TrackerGeomBuilderFromGeometricDet::getString") << "I need 1 "<< s << " tags";
	  abort();
	}
	return temp[0]; 
    }
    return "NotFound";
}

double TrackerGeomBuilderFromGeometricDet::getDouble(const std::string s,  DDExpandedView* ev){
  vector<std::string> temp;
  DDValue val(s);
  vector<const DDsvalues_type *> result = ev->specifics();
  vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it)
    {
      foundIt = DDfetch(*it,val);
      if (foundIt) break;
    }    
  if (foundIt)
    { 
      temp = val.strings(); 
      if (temp.size() != 1)
	{
	  edm::LogError("TrackerGeomBuilderFromGeometricDet::getDouble") << "I need 1 "<< s << " tags";
	  abort();
	}
      return double(atoi(temp[0].c_str())); 
    }
  return 0;
}
