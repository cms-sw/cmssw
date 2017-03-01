#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "Geometry/EcalTestBeam/test/ee/CaloGeometryLoaderTest.h"
#include "Geometry/EcalTestBeam/test/ee/CaloGeometryLoaderTest.icc"

template class CaloGeometryLoaderTest< EcalEndcapGeometry > ;

#include "DetectorDescription/Core/interface/DDFilteredView.h"


#include <iostream>
#include <vector>

using namespace std;



typedef CaloGeometryLoaderTest< EcalEndcapGeometry > EcalEGL ;

template <>
void 
EcalEGL::fillGeom( EcalEndcapGeometry*     geom ,
		   const EcalEGL::ParmVec& vv ,
		   const HepGeom::Transform3D&   tr ,
		   const DetId&            id     )
{
   std::vector<CCGFloat> pv ;
   pv.reserve( vv.size() ) ;
   for( unsigned int i ( 0 ) ; i != vv.size() ; ++i )
   {
      const double factor ( 1==i || 2==i || 6==i || 10==i ? 1 : k_ScaleFromDDDtoGeant ) ;
      pv.push_back( factor*vv[i] ) ;
   }

//   CaloCellGeometry::CornersVec corners ( geom->cornersMgr() ) ;
//   corners.resize() ;

   std::vector<GlobalPoint> corners ( 8 ) ;

   TruncatedPyramid::createCorners( pv, tr, corners ) ;
   const CCGFloat* parmPtr ( CaloCellGeometry::getParmPtr( pv, 
							   geom->parMgr(), 
							   geom->parVecVec() ) ) ;

   const GlobalPoint front ( 0.25*( corners[0].x() + 
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
   
   const GlobalPoint back  ( 0.25*( corners[4].x() + 
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

   geom->newCell( front, back, corners[0],
		  parmPtr, 
		  id ) ;
}

template <>
void 
EcalEGL::fillNamedParams( DDFilteredView      fv,
			  EcalEndcapGeometry* geom )
{
   bool doSubDets = fv.firstChild();
   while (doSubDets)
   {
      DDsvalues_type sv ( fv.mergedSpecifics() ) ;
	    
      //ncrys
      DDValue valNcrys("ncrys");
      if( DDfetch( &sv, valNcrys ) ) 
      {
	 const vector<double>& fvec = valNcrys.doubles();

	 // this parameter can only appear once
	 assert(fvec.size() == 1);
	 geom->setNumberOfCrystalPerModule((int)fvec[0]);
      }
      else 
	 continue;

      //nmods
      DDValue valNmods("nmods");
      if( DDfetch( &sv, valNmods ) ) 
      {
	 const vector<double>& fmvec = valNmods.doubles() ;

	 // there can only be one such value
	 assert(fmvec.size() == 1);
	 geom->setNumberOfModules((int)fmvec[0]);
      }
	    
      break;

      doSubDets = fv.nextSibling(); // go to next layer
   }
}
