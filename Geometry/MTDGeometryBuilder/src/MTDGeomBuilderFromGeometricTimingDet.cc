#include "Geometry/MTDGeometryBuilder/interface/MTDGeomBuilderFromGeometricTimingDet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopologyBuilder.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cfloat>
#include <cassert>
using std::vector;
using std::string;

namespace {
  void verifyDUinTG(MTDGeometry const & tg) {
    int off=0; int end=0;
    for ( int i=1; i!=2; i++) {
      auto det = i - 1;
      off = tg.offsetDU(det);
      end = tg.endsetDU(det); assert(end>=off); // allow empty subdetectors. Needed for upgrade
      for (int j=off; j!=end; ++j) {
	assert(tg.detUnits()[j]->geographicalId().subdetId()==i);
	assert(tg.detUnits()[j]->index()==j);
      }
    }
  }
}

MTDGeometry*
MTDGeomBuilderFromGeometricTimingDet::build( const GeometricTimingDet* gd, const PMTDParameters& ptp, const MTDTopology* tTopo )
{  
  theMTDDetTypeMap.clear();
   
  MTDGeometry* tracker = new MTDGeometry(gd);
  std::vector<const GeometricTimingDet*> comp;
  gd->deepComponents(comp);
  
  if(tTopo)  theTopo = tTopo;

  //define a vector which associate to the detid subdetector index -1 (from 0 to 5) the GeometridDet enumerator to be able to know which type of subdetector it is
  
  std::vector<GeometricTimingDet::GTDEnumType> gdsubdetmap(2,GeometricTimingDet::unknown); // hardcoded "2" should not be a surprise... 
  GeometricTimingDet::ConstGeometricTimingDetContainer subdetgd = gd->components();
  
  LogDebug("SubDetectorGeometricTimingDetType") 
    << "GeometricTimingDet enumerator values of the subdetectors" << std::endl;
  for(unsigned int i=0;i<subdetgd.size();++i) {
    MTDDetId mtdid(subdetgd[i]->geographicalId());
    assert(mtdid.mtdSubDetector()>0 && mtdid.mtdSubDetector()<3);
    gdsubdetmap[mtdid.mtdSubDetector()-1]= subdetgd[i]->type();
    LogTrace("SubDetectorGeometricTimingDetType") 
      << "subdet " << i 
      << " type " << subdetgd[i]->type()
      << " detid " << std::hex <<  subdetgd[i]->geographicalId().rawId() << std::dec
      << " subdetid " <<  subdetgd[i]->geographicalId().subdetId() << std::endl;
  }
  
  std::vector<const GeometricTimingDet*> dets[2];
  std::vector<const GeometricTimingDet*> & btl = dets[0]; btl.reserve(comp.size());
  std::vector<const GeometricTimingDet*> & etl = dets[1]; etl.reserve(comp.size());
 
  for(auto & i : comp) {
    MTDDetId mtdid(i->geographicalId());
    dets[mtdid.mtdSubDetector()-1].emplace_back(i);
  }
  
  //loop on all the six elements of dets and firstly check if they are from pixel-like detector and call buildPixel, then loop again and check if they are strip and call buildSilicon. "unknown" can be filled either way but the vector of GeometricTimingDet must be empty !!
  // this order is VERY IMPORTANT!!!!! For the moment I (AndreaV) understand that some pieces of code rely on pixel-like being before strip-like 
  
  // now building the Pixel-like subdetectors
  for(unsigned int i=0;i<2;++i) {
    if(gdsubdetmap[i] == GeometricTimingDet::BTL) 
      buildPixel(dets[i],tracker,
		 GeomDetEnumerators::SubDetector::TimingBarrel,
		 ptp); 
    if(gdsubdetmap[i] == GeometricTimingDet::ETL) 
      buildPixel(dets[i],tracker,
		 GeomDetEnumerators::SubDetector::TimingEndcap,
		 ptp);    
  }  
  
  buildGeomDet(tracker);//"GeomDet"

  verifyDUinTG(*tracker);
  
  return tracker;
}

void MTDGeomBuilderFromGeometricTimingDet::buildPixel(std::vector<const GeometricTimingDet*>  const & gdv, 
						      MTDGeometry* tracker,
						      GeomDetType::SubDetector det,
						      const PMTDParameters& ptp) // in y direction, cols. BIG_PIX_PER_ROC_Y = 0 for SLHC
{
  LogDebug("BuildingGeomDetUnits") 
    << " Pixel type. Size of vector: " << gdv.size() 
    << " GeomDetType subdetector: " << det 
    << " logical subdetector: " << GeomDetEnumerators::subDetGeom[det]
    << " big pix per ROC x: " << 0<< " y: " << 0
    << " is upgrade: " << true << std::endl;
  
  // this is a hack while we put things into the DDD
  int ROCrows(0),ROCcols(0),ROCSx(0),ROCSy(0);
  switch(det) {
  case GeomDetType::SubDetector::TimingBarrel:
    ROCrows = ptp.vitems_[0].vpars_[8];
    ROCcols = ptp.vitems_[0].vpars_[9];
    ROCSx   = ptp.vitems_[0].vpars_[10];
    ROCSy   = ptp.vitems_[0].vpars_[11];
    break;
  case GeomDetType::SubDetector::TimingEndcap:
    ROCrows = ptp.vitems_[1].vpars_[8];
    ROCcols = ptp.vitems_[1].vpars_[9];
    ROCSx   = ptp.vitems_[1].vpars_[10];
    ROCSy   = ptp.vitems_[1].vpars_[11];
    break;
    break;
  default:
    throw cms::Exception("UnknownDet") 
      << "MTDGeomBuilderFromGeometricTimingDet got a weird detector: " << det;
  }
  
  switch(det) {
  case GeomDetEnumerators::TimingBarrel:
    tracker->setOffsetDU(0);
    break;
  case GeomDetEnumerators::TimingEndcap:
    tracker->setOffsetDU(1);
    break;
  default:
    throw cms::Exception("MTDGeomBuilderFromGeometricTimingDet") << det << " is not a timing detector!";
  }

  for(auto i : gdv){

    std::string const & detName = i->name().fullname();
    if (theMTDDetTypeMap.find(detName) == theMTDDetTypeMap.end()) {
      std::unique_ptr<const Bounds> bounds(i->bounds());
      
      PixelTopology* t = 
	  MTDTopologyBuilder().build(&*bounds,
				     true,
				     ROCrows,
				     ROCcols,
				     0,0, // these are BIG_PIX_XXXXX
				     ROCSx, ROCSy);
      
      theMTDDetTypeMap[detName] = new MTDGeomDetType(t,detName,det);
      tracker->addType(theMTDDetTypeMap[detName]);
    }

    PlaneBuilderFromGeometricTimingDet::ResultType plane = buildPlaneWithMaterial(i);
    GeomDetUnit* temp =  new MTDGeomDetUnit(&(*plane),theMTDDetTypeMap[detName],i->geographicalID());

    tracker->addDetUnit(temp);
    tracker->addDetUnitId(i->geographicalID());
  }
  switch(det) {
  case GeomDetEnumerators::TimingBarrel:
    tracker->setEndsetDU(0);
    break;
  case GeomDetEnumerators::TimingEndcap:
    tracker->setEndsetDU(1);
    break;
  default:
    throw cms::Exception("MTDGeomBuilderFromGeometricTimingDet") << det << " is not a timing detector!";
  }
}

void MTDGeomBuilderFromGeometricTimingDet::buildGeomDet(MTDGeometry* tracker){

  auto const & gdu = tracker->detUnits();
  auto const & gduId = tracker->detUnitIds();

  for(u_int32_t i=0;i<gdu.size();i++){

    tracker->addDet(gdu[i]);
    tracker->addDetId(gduId[i]);
    string gduTypeName = gdu[i]->type().name();
    
  }
}

PlaneBuilderFromGeometricTimingDet::ResultType
MTDGeomBuilderFromGeometricTimingDet::buildPlaneWithMaterial(const GeometricTimingDet* gd,
							   double scale) const
{
  PlaneBuilderFromGeometricTimingDet planeBuilder;
  PlaneBuilderFromGeometricTimingDet::ResultType plane = planeBuilder.plane(gd);  
  //
  // set medium properties (if defined)
  //
  plane->setMediumProperties(MediumProperties(gd->radLength()*scale,gd->xi()*scale));

  return plane;
}
