#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"

CaloTowerGeometry::CaloTowerGeometry() {
}
  

CaloTowerGeometry::~CaloTowerGeometry() {}


unsigned int
CaloTowerGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isCaloTower() ) ;

   unsigned int index ( 0 ) ;// to be implemented

   return index ;
}

unsigned int
CaloTowerGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return 0 ;
}

std::vector<HepPoint3D> 
CaloTowerGeometry::localCorners( const double* pv,
				 unsigned int  i,
				 HepPoint3D&   ref )
{
   return ( calogeom::IdealObliquePrism::localCorners( pv, ref ) ) ;
}

CaloCellGeometry* 
CaloTowerGeometry::newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    CaloCellGeometry::CornersMgr* mgr,
			    const double*      parm ,
			    const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;

   assert( cgid.isCaloTower() ) ;

   return ( new calogeom::IdealObliquePrism( f1, mgr, parm ) ) ;
}
