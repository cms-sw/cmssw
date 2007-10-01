#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "Geometry/CaloGeometry/interface/CaloGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader< EcalEndcapGeometry > ;

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDNodes.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDName.h"
//#include "DetectorDescription/Core/interface/DDInit.h"
#include "DetectorDescription/Core/interface/DDScope.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDQuery.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>
#include <vector>

using namespace std;

typedef CaloGeometryLoader< EcalEndcapGeometry > EcalEGL ;


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
   geom->makeGridMap();
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
