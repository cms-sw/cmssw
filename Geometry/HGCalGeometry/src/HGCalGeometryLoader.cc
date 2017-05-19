#include "Geometry/HGCalGeometry/interface/HGCalGeometryLoader.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define EDM_ML_DEBUG

typedef CaloCellGeometry::CCGFloat CCGFloat;
typedef std::vector<float> ParmVec;

HGCalGeometryLoader::HGCalGeometryLoader () {}
HGCalGeometryLoader::~HGCalGeometryLoader () {}

HGCalGeometry* HGCalGeometryLoader::build (const HGCalTopology& topology) {

  // allocate geometry
  HGCalGeometry* geom = new HGCalGeometry (topology);
  unsigned int numberOfCells = topology.totalGeomModules(); // both sides
  unsigned int numberExpected= topology.allGeomModules();
#ifdef EDM_ML_DEBUG
  std::cout << "Number of Cells " << numberOfCells << ":" << numberExpected
	    << " for sub-detector " << topology.subDetector() 
	    << " Shape parameters " << HGCalGeometry::k_NumberOfShapes << ":" 
	    << HGCalGeometry::k_NumberOfParametersPerShape << std::endl;
#endif
  geom->allocateCorners( numberOfCells ) ;
  geom->allocatePar(HGCalGeometry::k_NumberOfShapes,
		    HGCalGeometry::k_NumberOfParametersPerShape);
  ForwardSubdetector subdet  = topology.subDetector();
  bool               detType = topology.detectorType();

  // loop over modules
  ParmVec params(HGCalGeometry::k_NumberOfParametersPerShape,0);
  unsigned int counter(0);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeometryLoader with # of transformation matrices " 
	    << topology.dddConstants().getTrFormN() << " and "
	    << topology.dddConstants().volumes() << ":"
	    << topology.dddConstants().sectors() << " volumes" << std::endl;
#endif
  for (unsigned itr=0; itr<topology.dddConstants().getTrFormN(); ++itr) {
    HGCalParameters::hgtrform mytr = topology.dddConstants().getTrForm(itr);
    int zside  = mytr.zp;
    int layer  = mytr.lay;
#ifdef EDM_ML_DEBUG
    unsigned int kount(0);
    std::cout << "HGCalGeometryLoader:: Z:Layer " << zside << ":" << layer 
	      << std::endl;
#endif
    if (topology.geomMode() == HGCalGeometryMode::Square) {
      int sector = mytr.sec;
      int subSec = (detType ? mytr.subsec : 0);
      const HepGeom::Transform3D ht3d (mytr.hr, mytr.h3v);
      DetId detId= ((subdet ==  HGCEE) ?
		    (DetId)(HGCEEDetId(subdet,zside,layer,sector,subSec,0)) :
		    (DetId)(HGCHEDetId(subdet,zside,layer,sector,subSec,0)));
#ifdef EDM_ML_DEBUG
      std::cout << "HGCalGeometryLoader:: Sector:Subsector " << sector << ":" 
		<< subSec << " transf " << ht3d.getTranslation() << " and " 
		<< ht3d.getRotation();
#endif
      for (unsigned int k=0; k<topology.dddConstants().volumes(); ++k) {
	HGCalParameters::hgtrap vol = topology.dddConstants().getModule(k,false,true);
	if (vol.lay == layer) {
	  double alpha = ((detType && subSec == 0) ? -fabs(vol.alpha) :
			  fabs(vol.alpha));
	  params[0] = vol.dz;
	  params[1] = params[2] = 0;
	  params[3] = params[7] = vol.h;
	  params[4] = params[8] = vol.bl;
	  params[5] = params[9] = vol.tl;
	  params[6] = params[10]= alpha;
	  params[11]= vol.cellSize;
	  buildGeom(params, ht3d, detId, geom);
	  counter++;
#ifdef EDM_ML_DEBUG
	  ++kount;
#endif
	  break;
	}
      }
    } else {
      for (int wafer=0; wafer<topology.dddConstants().sectors(); ++wafer) {
	if (topology.dddConstants().waferInLayer(wafer,layer,true)) {
	  int type = topology.dddConstants().waferTypeT(wafer);
	  if (type != 1) type = 0;
	  DetId detId = (DetId)(HGCalDetId(subdet,zside,layer,type,wafer,0));
	  std::pair<double,double>  w = topology.dddConstants().waferPosition(wafer);
	  double xx = (zside > 0) ? w.first : -w.first;
	  CLHEP::Hep3Vector h3v(xx,w.second,mytr.h3v.z());
	  const HepGeom::Transform3D ht3d (mytr.hr, h3v);
#ifdef EDM_ML_DEBUG
	  std::cout << "HGCalGeometryLoader:: Wafer:Type " << wafer << ":" 
		    << type << " DetId " << HGCalDetId(detId) << std::hex
		    << " " << detId.rawId() << std::dec << " transf " 
		    << ht3d.getTranslation() << " and " << ht3d.getRotation();
#endif
	  HGCalParameters::hgtrap vol = topology.dddConstants().getModule(wafer,true,true);
	  params[0] = vol.dz;
	  params[1] = params[2] = 0;
	  params[3] = params[7] = vol.h;
	  params[4] = params[8] = vol.bl;
	  params[5] = params[9] = vol.tl;
	  params[6] = params[10]= 0;
	  params[11]= topology.dddConstants().cellSizeHex(type);

	  buildGeom(params, ht3d, detId, geom);
	  counter++;
#ifdef EDM_ML_DEBUG
	  ++kount;
#endif
	}
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << kount << " modules found in Layer " << layer << " Z "
	      << zside << std::endl;
#endif
  }

  geom->sortDetIds();

  if (counter != numberExpected) {
    std::cerr << "Inconsistent # of cells: expected " << numberExpected << ":"
	      << numberOfCells << " , inited " << counter << std::endl;
    assert( counter == numberOfCells ) ;
  }

  return geom;
}

void HGCalGeometryLoader::buildGeom(const ParmVec& params, 
				    const HepGeom::Transform3D& ht3d, 
				    const DetId& detId,  HGCalGeometry* geom) {

#ifdef EDM_ML_DEBUG
  std::cout << "Volume Parameters";
  for (unsigned int i=0; i<12; ++i) std::cout << " : " << params[i];
  std::cout << std::endl;
#endif
  std::vector<GlobalPoint> corners (8);
	
  FlatTrd::createCorners( params, ht3d, corners ) ;
	
  const CCGFloat* parmPtr (CaloCellGeometry::getParmPtr(params, 
							geom->parMgr(), 
							geom->parVecVec() ) ) ;
	
  GlobalPoint front ( 0.25*( corners[0].x() + 
			     corners[1].x() + 
			     corners[2].x() + 
			     corners[3].x()   ),
		      0.25*( corners[0].y() + 
			     corners[1].y() + 
			     corners[2].y() + 
			     corners[3].y()   ),
		      0.25*( corners[0].z() + 
			     corners[1].z() + 
			     corners[2].z() + 
			     corners[3].z()   ) ) ;
  
  GlobalPoint back  ( 0.25*( corners[4].x() + 
			     corners[5].x() + 
			     corners[6].x() + 
			     corners[7].x()   ),
		      0.25*( corners[4].y() + 
			     corners[5].y() + 
			     corners[6].y() + 
			     corners[7].y()   ),
		      0.25*( corners[4].z() + 
			     corners[5].z() + 
			     corners[6].z() + 
			     corners[7].z()   ) ) ;
  
  if (front.mag2() > back.mag2()) { // front should always point to the center, so swap front and back
    std::swap (front, back);
    std::swap_ranges (corners.begin(), corners.begin()+4, corners.begin()+4); 
  }
	
  geom->newCell(front, back, corners[0], parmPtr, detId) ;
}
