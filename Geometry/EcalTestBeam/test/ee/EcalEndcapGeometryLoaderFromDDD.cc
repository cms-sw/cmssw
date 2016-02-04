#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "Geometry/EcalTestBeam/test/ee/CaloGeometryLoaderTest.h"
#include "Geometry/EcalTestBeam/test/ee/CaloGeometryLoaderTest.icc"

template class CaloGeometryLoaderTest< EcalEndcapGeometry > ;

#include "DetectorDescription/Core/interface/DDFilteredView.h"
//#include "DetectorDescription/Core/interface/DDInit.h"


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
   std::vector<double> pv ;
   pv.reserve( vv.size() ) ;
   for( unsigned int i ( 0 ) ; i != vv.size() ; ++i )
   {
      const double factor ( 1==i || 2==i || 6==i || 10==i ? 1 : k_ScaleFromDDDtoGeant ) ;
      pv.push_back( factor*vv[i] ) ;
   }

   CaloCellGeometry::CornersVec corners ( geom->cornersMgr() ) ;
   corners.resize() ;

   TruncatedPyramid::createCorners( pv, tr, corners ) ;
   const double* parmPtr ( CaloCellGeometry::getParmPtr( pv, 
							 geom->parMgr(), 
							 geom->parVecVec() ) ) ;

   TruncatedPyramid* cell ( new TruncatedPyramid( corners , parmPtr ) ) ;

   geom->addCell( id, cell );
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
