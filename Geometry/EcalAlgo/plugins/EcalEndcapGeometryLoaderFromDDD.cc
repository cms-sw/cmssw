#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader< EcalEndcapGeometry > ;

#include "DetectorDescription/Core/interface/DDFilteredView.h"
//#include "DetectorDescription/Core/interface/DDInit.h"


#include <iostream>
#include <vector>

using namespace std;

typedef CaloGeometryLoader< EcalEndcapGeometry > EcalEGL ;


template <>
unsigned int 
EcalEGL::whichTransform( const DetId& id ) const
{
   const EEDetId eeid ( id ) ;
   const int ix ( eeid.ix() ) ;
   return ( ix/51 + ( eeid.zside()<0 ? 0 : 2 ) ) ;
}

template <>
void 
EcalEGL::fillGeom( EcalEndcapGeometry*     geom ,
		   const EcalEGL::ParmVec& pv ,
		   const HepTransform3D&   tr ,
		   const DetId&            id     )
{
   CaloCellGeometry::CornersVec corners ( geom->cornersMgr() ) ;
   corners.resize() ;

   TruncatedPyramid::createCorners( pv, tr, corners ) ;

   TruncatedPyramid* cell ( new TruncatedPyramid( corners ) ) ;

   geom->addCell( id, cell );
}

template <>
void 
EcalEGL::extraStuff( EcalEndcapGeometry* geom )
{
   geom->initialize();
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
