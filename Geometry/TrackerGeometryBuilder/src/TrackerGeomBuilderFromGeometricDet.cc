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
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"


#include <cfloat>
#include <vector>

using std::vector;
using std::string;


TrackerGeometry* TrackerGeomBuilderFromGeometricDet::build(const DDCompactView* cpv, const GeometricDet* gd){

  DDExpandedView ev(*cpv);

  TrackerGeometry* tracker = new TrackerGeometry();
  std::vector<const GeometricDet*> const & comp = gd->deepComponents();

  std::vector<const GeometricDet*> pixB; pixB.reserve(comp.size());
  std::vector<const GeometricDet*> pixF; pixF.reserve(comp.size());
  std::vector<const GeometricDet*> tib;  tib.reserve(comp.size());
  std::vector<const GeometricDet*> tid;  tid.reserve(comp.size());
  std::vector<const GeometricDet*> tob;  tob.reserve(comp.size());
  std::vector<const GeometricDet*> tec;  tec.reserve(comp.size());

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
  buildPixel(pixB,&ev,tracker,theDetIdToEnum.type(1), "barrel"); //"PixelBarrel" 
  buildPixel(pixF,&ev,tracker,theDetIdToEnum.type(2), "endcap"); //"PixelEndcap" 
  buildSilicon(tib,&ev,tracker,theDetIdToEnum.type(3), "barrel");// "TIB"	
  buildSilicon(tid,&ev,tracker,theDetIdToEnum.type(4), "endcap");//"TID" 
  buildSilicon(tob,&ev,tracker,theDetIdToEnum.type(5), "barrel");//"TOB"	
  buildSilicon(tec,&ev,tracker,theDetIdToEnum.type(6), "endcap");//"TEC"        
  buildGeomDet(tracker);//"GeomDet"
  return tracker;
}

void TrackerGeomBuilderFromGeometricDet::buildPixel(std::vector<const GeometricDet*>  const & gdv, 
						    DDExpandedView* ev,
						    TrackerGeometry* tracker,
						    GeomDetType::SubDetector& det,
						    const std::string& part){ 

  static std::map<std::string,PixelGeomDetType*> detTypeMap;

  for(u_int32_t i=0; i<gdv.size(); i++){

    ev->goTo(gdv[i]->navType());
    std::string const & detName = gdv[i]->name();
    if (detTypeMap.find(detName) == detTypeMap.end()) {

      PixelTopology* t = 
	theTopologyBuilder->buildPixel(gdv[i]->bounds(),
				       getDouble("PixelROCRows",ev),
				       getDouble("PixelROCCols",ev),
				       getDouble("PixelROC_X"  ,ev),
				       getDouble("PixelROC_Y"  ,ev),
				       part);
      
      detTypeMap[detName] = new PixelGeomDetType(t,detName,det);
      tracker->addType(detTypeMap[detName]);
    }

    PlaneBuilderFromGeometricDet::ResultType plane = buildPlaneWithMaterial(gdv[i], ev);
    GeomDetUnit* temp =  new PixelGeomDetUnit(&(*plane),detTypeMap[detName],gdv[i]);

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());
  }
}

void TrackerGeomBuilderFromGeometricDet::buildSilicon(std::vector<const GeometricDet*>  const & gdv, 
						      DDExpandedView* ev,
						      TrackerGeometry* tracker,
						      GeomDetType::SubDetector& det,
						      const std::string& part)
{ 
  static std::map<std::string,StripGeomDetType*> detTypeMap;
  
  for(u_int32_t i=0;i<gdv.size();i++){

    ev->goTo(gdv[i]->navType());
    std::string const & detName = gdv[i]->name();
    if (detTypeMap.find(detName) == detTypeMap.end()) {

      bool stereo = false;
      if(getString("TrackerStereoDetectors",ev)=="true"){
	stereo = true;
      }
      StripTopology* t =
	theTopologyBuilder->buildStrip(gdv[i]->bounds(),
				       getDouble("SiliconAPVNumber",ev),
				       part);
      detTypeMap[detName] = new  StripGeomDetType( t,detName,det,
						   stereo);
      tracker->addType(detTypeMap[detName]);
    }
    PlaneBuilderFromGeometricDet::ResultType plane = buildPlaneWithMaterial(gdv[i],ev);  
    GeomDetUnit* temp = new StripGeomDetUnit(&(*plane), detTypeMap[detName],gdv[i]);
    
    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());
  }  
}


void TrackerGeomBuilderFromGeometricDet::buildGeomDet(TrackerGeometry* tracker){
  PlaneBuilderForGluedDet gluedplaneBuilder;
  std::vector<GeomDetUnit*> const & gdu= tracker->detUnits();
  std::vector<DetId> const & gduId = tracker->detUnitIds();
  std::vector<const GeomDetUnit *> glued; glued.reserve(2);

  for(u_int32_t i=0;i<gdu.size();i++){
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
	throw cms::Exception("Configuration") <<"No partner detector found \n"
					<<"There is a problem on Tracker geometry configuration\n";
      }
      const GeomDetUnit* dum = gdu[partner_pos];
      glued.clear();
      glued.push_back(dum);
      glued.push_back(dus);
      PlaneBuilderForGluedDet::ResultType plane = gluedplaneBuilder.plane(glued);
      GluedGeomDet* gluedDet = new GluedGeomDet(&(*plane),dum,dus);
      tracker->addDet((GeomDet*) gluedDet);
      tracker->addDetId(DetId(sidet.glued()));
    }
  }
}


std::string TrackerGeomBuilderFromGeometricDet::getString(const std::string & s, DDExpandedView* ev) const
{
    DDValue val(s);
    vector<const DDsvalues_type *> result;
    ev->specificsV(result);
    vector<const DDsvalues_type *>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it)
    {
	foundIt = DDfetch(*it,val);
	if (foundIt) break;

    }    
    if (foundIt)
    { 
	const std::vector<std::string> & temp = val.strings(); 
	if (temp.size() != 1)
	{
	  throw cms::Exception("Configuration") << "I need 1 "<< s << " tags";
	}
	return temp[0]; 
    }
    return "NotFound";
}

double TrackerGeomBuilderFromGeometricDet::getDouble(const std::string & s,  DDExpandedView* ev) const 
{
  DDValue val(s);
  vector<const DDsvalues_type *> result;
  ev->specificsV(result);
  vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it)
    {
      foundIt = DDfetch(*it,val);
      if (foundIt) break;
    }    
  if (foundIt)
    { 
      const std::vector<std::string> & temp = val.strings(); 
      if (temp.size() != 1)
	{
	  throw cms::Exception("Configuration") << "I need 1 "<< s << " tags";
	}
      return double(atof(temp[0].c_str())); 
    }
  return 0;
}

PlaneBuilderFromGeometricDet::ResultType
TrackerGeomBuilderFromGeometricDet::buildPlaneWithMaterial(const GeometricDet* gd, 
							   DDExpandedView* ev) const
{
  PlaneBuilderFromGeometricDet planeBuilder;
  PlaneBuilderFromGeometricDet::ResultType plane = planeBuilder.plane(gd);  
  //
  // set medium properties (if defined)
  //
  double radLength = getDouble("TrackerRadLength",ev);
  double xi = getDouble("TrackerXi",ev);

  plane->setMediumProperties( new MediumProperties(radLength,xi) );

  return plane;
}
