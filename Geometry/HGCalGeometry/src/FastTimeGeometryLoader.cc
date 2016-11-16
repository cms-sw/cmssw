#include "Geometry/HGCalGeometry/interface/FastTimeGeometryLoader.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#define EDM_ML_DEBUG

typedef CaloCellGeometry::CCGFloat CCGFloat;
typedef std::vector<float> ParmVec;

FastTimeGeometryLoader::FastTimeGeometryLoader () {}
FastTimeGeometryLoader::~FastTimeGeometryLoader () {}

FastTimeGeometry* FastTimeGeometryLoader::build (const FastTimeTopology& topology) {

  // allocate geometry
  FastTimeGeometry* geom = new FastTimeGeometry (topology);
  unsigned int numberOfCells = topology.totalGeomModules(); // both sides
  int                detType = topology.detectorType();
#ifdef EDM_ML_DEBUG
  std::cout << "Number of Cells " << numberOfCells <<  " for type " << detType
	    << " of sub-detector " << topology.subDetector() 
	    << " Shape parameters " << FastTimeGeometry::k_NumberOfShapes
	    << ":"  << FastTimeGeometry::k_NumberOfParametersPerShape 
	    << std::endl;
#endif
  geom->allocateCorners( numberOfCells ) ;
  geom->allocatePar(FastTimeGeometry::k_NumberOfShapes,
		    FastTimeGeometry::k_NumberOfParametersPerShape);

  // loop over modules
  ParmVec params(FastTimeGeometry::k_NumberOfParametersPerShape,0);
  unsigned int counter(0);
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeGeometryLoader with # of transformation matrices " 
	    << numberOfCells << std::endl;
#endif
  for (unsigned itr=0; itr<numberOfCells; ++itr) {
    int zside  = (itr == 0) ? 1 : -1;
#ifdef EDM_ML_DEBUG
    std::cout << "FastTimeGeometryLoader:: Z:Layer:Type " << zside
	      << ":" << detType <<std::endl;
#endif
    double zv = zside*(topology.dddConstants().getZPos(detType));
    const CLHEP::HepRep3x3 rotation = (zside > 0) ?
      CLHEP::HepRep3x3(1,0,0,0,1,0,0,0,1) :
      CLHEP::HepRep3x3(-1,0,0,0,1,0,0,0,-1);
    const CLHEP::HepRotation hr ( rotation );
    const CLHEP::Hep3Vector h3v(0,0,zv);
    const HepGeom::Transform3D ht3d (hr, h3v);
    DetId detId = (DetId)(FastTimeDetId(detType,0,0,zside));
#ifdef EDM_ML_DEBUG
    std::cout << "FastTimeGeometryLoader:: transf " << ht3d.getTranslation() 
	      << " and " << ht3d.getRotation();
#endif
    params[0] = topology.dddConstants().getZHalf(detType);
    params[1] = params[2] = 0;
    params[3] = params[7] = topology.dddConstants().getRin(detType);
    params[4] = params[8] = topology.dddConstants().getRout(detType);
    params[5] = params[9] = topology.dddConstants().getRout(detType);
    params[6] = params[10]= 0;
    params[11]= zside;
    buildGeom(params, ht3d, detId, topology, geom);
    counter++;
  }

  geom->sortDetIds();

  if (counter != numberOfCells) {
    std::cerr << "inconsistent # of cells: expected " << numberOfCells 
	      << " , inited " << counter << std::endl;
    assert( counter == numberOfCells ) ;
  }
  
  return geom;
}

void FastTimeGeometryLoader::buildGeom(const ParmVec& params, 
				       const HepGeom::Transform3D& ht3d, 
				       const DetId& detId, 
				       const FastTimeTopology& topology,
				       FastTimeGeometry* geom) {

#ifdef EDM_ML_DEBUG
  std::cout << "Volume Parameters";
  for (unsigned int i=0; i<12; ++i) std::cout << " : " << params[i];
  std::cout << std::endl;
#endif
  FastTimeDetId id = FastTimeDetId(detId);
  std::vector<GlobalPoint> corners = topology.dddConstants().getCorners(id.type(),1,1,id.zside());
	
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
