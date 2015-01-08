#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderForGluedDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripTopologyBuilder.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cfloat>
#include <cassert>
using std::vector;
using std::string;

namespace {
  void verifyDUinTG(TrackerGeometry const & tg) {
    int off=0; int end=0;
    for ( int i=1; i!=7; i++) {
      auto det = GeomDetEnumerators::tkDetEnum[i];
      off = tg.offsetDU(det);
      end = tg.endsetDU(det); assert(end>=off); // allow empty subdetectors. Needed for upgrade
      for (int j=off; j!=end; ++j) {
	assert(tg.detUnits()[j]->geographicalId().subdetId()==i);
	assert(GeomDetEnumerators::subDetGeom[tg.detUnits()[j]->subDetector()]==det);
	assert(tg.detUnits()[j]->index()==j);
      }
    }
  }
}

TrackerGeometry*
TrackerGeomBuilderFromGeometricDet::build( const GeometricDet* gd, const edm::ParameterSet& pSet )
{
  bool upgradeGeometry = false;
  int BIG_PIX_PER_ROC_X = 1;
  int BIG_PIX_PER_ROC_Y = 2;
  
  if( pSet.exists( "trackerGeometryConstants" ))
  {
    const edm::ParameterSet tkGeomConsts( pSet.getParameter<edm::ParameterSet>( "trackerGeometryConstants" ));
    upgradeGeometry = tkGeomConsts.getParameter<bool>( "upgradeGeometry" );  
    BIG_PIX_PER_ROC_X = tkGeomConsts.getParameter<int>( "BIG_PIX_PER_ROC_X" );
    BIG_PIX_PER_ROC_Y = tkGeomConsts.getParameter<int>( "BIG_PIX_PER_ROC_Y" );
  }
    
  thePixelDetTypeMap.clear();
  theStripDetTypeMap.clear();
   
  TrackerGeometry* tracker = new TrackerGeometry(gd);
  std::vector<const GeometricDet*> comp;
  gd->deepComponents(comp);

  //define a vector which associate to the detid subdetector index -1 (from 0 to 5) the GeometridDet enumerator to be able to know which type of subdetector it is
  
  std::vector<GeometricDet::GDEnumType> gdsubdetmap(6,GeometricDet::unknown); // hardcoded "6" should not be a surprise... 
  GeometricDet::ConstGeometricDetContainer subdetgd = gd->components();
  
  LogDebug("SubDetectorGeometricDetType") << "GeometriDet enumerator values of the subdetectors";
  for(unsigned int i=0;i<subdetgd.size();++i) {
    assert(subdetgd[i]->geographicalId().subdetId()>0 && subdetgd[i]->geographicalId().subdetId()<7);
    gdsubdetmap[subdetgd[i]->geographicalId().subdetId()-1]= subdetgd[i]->type();
    LogTrace("SubDetectorGeometricDetType") << "subdet " << i 
					    << " type " << subdetgd[i]->type()
					    << " detid " <<  subdetgd[i]->geographicalId()
					    << " subdetid " <<  subdetgd[i]->geographicalId().subdetId();
  }
  
  std::vector<const GeometricDet*> dets[6];
  std::vector<const GeometricDet*> & pixB = dets[0]; pixB.reserve(comp.size());
  std::vector<const GeometricDet*> & pixF = dets[1]; pixF.reserve(comp.size());
  std::vector<const GeometricDet*> & tib  = dets[2];  tib.reserve(comp.size());
  std::vector<const GeometricDet*> & tid  = dets[3];  tid.reserve(comp.size());
  std::vector<const GeometricDet*> & tob  = dets[4];  tob.reserve(comp.size());
  std::vector<const GeometricDet*> & tec  = dets[5];  tec.reserve(comp.size());

  for(u_int32_t i = 0;i<comp.size();i++)
    dets[comp[i]->geographicalID().subdetId()-1].push_back(comp[i]);
  
  //loop on all the six elements of dets and firstly check if they are from pixel-like detector and call buildPixel, then loop again and check if they are strip and call buildSilicon. "unknown" can be filled either way but the vector of GeometricDet must be empty !!
  // this order is VERY IMPORTANT!!!!! For the moment I (AndreaV) understand that some pieces of code rely on pixel-like being before strip-like 
  
  // now building the Pixel-like subdetectors
  for(unsigned int i=0;i<6;++i) {
    if(gdsubdetmap[i] == GeometricDet::PixelBarrel) 
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::PixelBarrel,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
    if(gdsubdetmap[i] == GeometricDet::PixelPhase1Barrel) 
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::P1PXB,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
    if(gdsubdetmap[i] == GeometricDet::PixelEndCap)
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::PixelEndcap,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
    if(gdsubdetmap[i] == GeometricDet::PixelPhase1EndCap)
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::P1PXEC,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
    if(gdsubdetmap[i] == GeometricDet::PixelPhase2EndCap)
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::P2PXEC,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
    if(gdsubdetmap[i] == GeometricDet::OTPhase2Barrel) 
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::P2OTB,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
    if(gdsubdetmap[i] == GeometricDet::OTPhase2EndCap) 
      buildPixel(dets[i],tracker,GeomDetEnumerators::SubDetector::P2OTEC,
		 upgradeGeometry,
		 BIG_PIX_PER_ROC_X,
		 BIG_PIX_PER_ROC_Y); 
  }
  //now building Strips
  for(unsigned int i=0;i<6;++i) {
    if(gdsubdetmap[i] == GeometricDet::TIB)   buildSilicon(dets[i],tracker,GeomDetEnumerators::SubDetector::TIB, "barrel");
    if(gdsubdetmap[i] == GeometricDet::TID)   buildSilicon(dets[i],tracker,GeomDetEnumerators::SubDetector::TID, "endcap");
    if(gdsubdetmap[i] == GeometricDet::TOB)   buildSilicon(dets[i],tracker,GeomDetEnumerators::SubDetector::TOB, "barrel");
    if(gdsubdetmap[i] == GeometricDet::TEC)   buildSilicon(dets[i],tracker,GeomDetEnumerators::SubDetector::TEC, "endcap");
  }  
  // and finally the "empty" subdetectors (maybe it is not needed)
  for(unsigned int i=0;i<6;++i) {
    if(gdsubdetmap[i] == GeometricDet::unknown) {
      if(dets[i].size()!=0) throw cms::Exception("NotEmptyUnknownSubDet") << "Subdetector " << i+1 << " is unknown but it is not empty: " << dets[i].size();
      buildSilicon(dets[i],tracker,GeomDetEnumerators::tkDetEnum[i+1], "barrel"); // "barrel" is used but it is irrelevant
    }
  }
  buildGeomDet(tracker);//"GeomDet"

  verifyDUinTG(*tracker);

  return tracker;
}

void TrackerGeomBuilderFromGeometricDet::buildPixel(std::vector<const GeometricDet*>  const & gdv, 
						    TrackerGeometry* tracker,
						    GeomDetType::SubDetector det,
						    bool upgradeGeometry,
						    int BIG_PIX_PER_ROC_X, // in x direction, rows. BIG_PIX_PER_ROC_X = 0 for SLHC
						    int BIG_PIX_PER_ROC_Y) // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
{
  LogDebug("BuildingGeomDetUnits") << " Pixel type. Size of vector: " << gdv.size() 
				   << " GeomDetType subdetector: " << det 
				   << " logical subdetector: " << GeomDetEnumerators::subDetGeom[det]
				   << " big pix per ROC x: " << BIG_PIX_PER_ROC_X << " y: " << BIG_PIX_PER_ROC_Y
				   << " is upgrade: " << upgradeGeometry;
  
  tracker->setOffsetDU(GeomDetEnumerators::subDetGeom[det]);

  for(u_int32_t i=0; i<gdv.size(); i++){

    std::string const & detName = gdv[i]->name().fullname();
    if (thePixelDetTypeMap.find(detName) == thePixelDetTypeMap.end()) {
      std::unique_ptr<const Bounds> bounds(gdv[i]->bounds());
      
      PixelTopology* t = 
	  PixelTopologyBuilder().build(&*bounds,
				       upgradeGeometry,
				       gdv[i]->pixROCRows(),
				       gdv[i]->pixROCCols(),
				       BIG_PIX_PER_ROC_X,
				       BIG_PIX_PER_ROC_Y,
				       gdv[i]->pixROCx(), gdv[i]->pixROCy());
      
      thePixelDetTypeMap[detName] = new PixelGeomDetType(t,detName,det);
      tracker->addType(thePixelDetTypeMap[detName]);
    }

    PlaneBuilderFromGeometricDet::ResultType plane = buildPlaneWithMaterial(gdv[i]);
    GeomDetUnit* temp =  new PixelGeomDetUnit(&(*plane),thePixelDetTypeMap[detName],gdv[i]->geographicalID());

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());
  }
  tracker->setEndsetDU(GeomDetEnumerators::subDetGeom[det]);
}

void TrackerGeomBuilderFromGeometricDet::buildSilicon(std::vector<const GeometricDet*>  const & gdv, 
						      TrackerGeometry* tracker,
						      GeomDetType::SubDetector det,
						      const std::string& part)
{ 
  LogDebug("BuildingGeomDetUnits") << " Strip type. Size of vector: " << gdv.size() 
				   << " GeomDetType subdetector: " << det 
				   << " logical subdetector: " << GeomDetEnumerators::subDetGeom[det]
				   << " part " << part;
  
  tracker->setOffsetDU(GeomDetEnumerators::subDetGeom[det]);

  for(u_int32_t i=0;i<gdv.size();i++){

    std::string const & detName = gdv[i]->name().fullname();
    if (theStripDetTypeMap.find(detName) == theStripDetTypeMap.end()) {
       std::unique_ptr<const Bounds> bounds(gdv[i]->bounds());
       StripTopology* t =
	   StripTopologyBuilder().build(&*bounds,
				       gdv[i]->siliconAPVNum(),
				       part);
      theStripDetTypeMap[detName] = new  StripGeomDetType( t,detName,det,
						   gdv[i]->stereo());
      tracker->addType(theStripDetTypeMap[detName]);
    }
     
    StripSubdetector sidet( gdv[i]->geographicalID());
    double scale  = (sidet.partnerDetId()) ? 0.5 : 1.0 ;	

    PlaneBuilderFromGeometricDet::ResultType plane = buildPlaneWithMaterial(gdv[i],scale);  
    GeomDetUnit* temp = new StripGeomDetUnit(&(*plane), theStripDetTypeMap[detName],gdv[i]->geographicalID());
    
    tracker->addDetUnit(temp);
    tracker->addDetUnitId(gdv[i]->geographicalID());
  }  
  tracker->setEndsetDU(GeomDetEnumerators::subDetGeom[det]);

}


void TrackerGeomBuilderFromGeometricDet::buildGeomDet(TrackerGeometry* tracker){
  PlaneBuilderForGluedDet gluedplaneBuilder;
  auto  const & gdu= tracker->detUnits();
  auto  const & gduId = tracker->detUnitIds();

  for(u_int32_t i=0;i<gdu.size();i++){
    StripSubdetector sidet( gduId[i].rawId());
    tracker->addDet((GeomDet*) gdu[i]);
    tracker->addDetId(gduId[i]);      
    if(sidet.glued()!=0&&sidet.stereo()==1){
      int partner_pos=-1;
      for(u_int32_t jj=0;jj<gduId.size();jj++){
	if(sidet.partnerDetId()== gduId[jj]) {
	  partner_pos=jj;
	  break;
	}
      }
      const GeomDetUnit* dus = gdu[i];
      if(partner_pos==-1){
	throw cms::Exception("Configuration") <<"No partner detector found \n"
					<<"There is a problem on Tracker geometry configuration\n";
      }
      const GeomDetUnit* dum = gdu[partner_pos];
      std::vector<const GeomDetUnit *> glued(2);
      glued[0]=dum;
      glued[1]=dus;
      PlaneBuilderForGluedDet::ResultType plane = gluedplaneBuilder.plane(glued);
      GluedGeomDet* gluedDet = new GluedGeomDet(&(*plane),dum,dus);
      tracker->addDet((GeomDet*) gluedDet);
      tracker->addDetId(DetId(sidet.glued()));
    }
  }
}

PlaneBuilderFromGeometricDet::ResultType
TrackerGeomBuilderFromGeometricDet::buildPlaneWithMaterial(const GeometricDet* gd,
							   double scale) const
{
  PlaneBuilderFromGeometricDet planeBuilder;
  PlaneBuilderFromGeometricDet::ResultType plane = planeBuilder.plane(gd);  
  //
  // set medium properties (if defined)
  //
  plane->setMediumProperties(MediumProperties(gd->radLength()*scale,gd->xi()*scale));

  return plane;
}
