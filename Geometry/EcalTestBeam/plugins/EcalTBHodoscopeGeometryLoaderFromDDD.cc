#include "Geometry/EcalTestBeam/plugins/EcalTBHodoscopeGeometryLoaderFromDDD.h"

#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDName.h"

typedef CaloCellGeometry::CCGFloat CCGFloat ;

#include <iostream>
#include <vector>

std::auto_ptr<CaloSubdetectorGeometry> 
EcalTBHodoscopeGeometryLoaderFromDDD::load( const DDCompactView* cpv ) 
{
   std::cout << "[EcalTBHodoscopeGeometryLoaderFromDDD]:: start the construction of EcalTBHodoscope" << std::endl;

   std::auto_ptr<CaloSubdetectorGeometry> ebg
      ( new EcalTBHodoscopeGeometry() ) ;

   makeGeometry( cpv, ebg.get() ) ;

   std::cout << "[EcalTBHodoscopeGeometryLoaderFromDDD]:: Returning EcalTBHodoscopeGeometry" << std::endl;

   return ebg;
}

void 
EcalTBHodoscopeGeometryLoaderFromDDD::makeGeometry(
   const DDCompactView*     cpv ,
   CaloSubdetectorGeometry* ebg  )
{
   if( ebg->cornersMgr() == 0 ) ebg->allocateCorners( CaloTowerDetId::kSizeForDenseIndexing ) ;
   if( ebg->parMgr()     == 0 ) ebg->allocatePar( 10, 3 ) ;
  
   DDFilter* filter = getDDFilter();

   DDFilteredView fv(*cpv);
   fv.addFilter(*filter);
  
   bool doSubDets;
   for (doSubDets = fv.firstChild(); doSubDets ; doSubDets = fv.nextSibling())
   {
      
#if 0
      std::string answer = getDDDString("ReadOutName",&fv);
      if (answer != "EcalTBH4BeamHits")
        continue;
#endif
      
      const DDSolid & solid = fv.logicalPart().solid();

      if( solid.shape() != ddbox ) 
      {
	 throw cms::Exception("DDException") << std::string(__FILE__) 
			    << "\n CaloGeometryEcalTBHodoscope::upDate(...): currently only box fiber shapes supported ";
	 edm::LogWarning("EcalTBHodoscopeGeometry") << "Wrong shape for sensitive volume!" << solid;
      }
       
      std::vector<double> pv = solid.parameters();      

      // use preshower strip as box in space representation

      // rotate the box and then move it
      DD3Vector x, y, z;
      fv.rotation().GetComponents(x,y,z);
      CLHEP::Hep3Vector hx(x.X(), x.Y(), x.Z());
      CLHEP::Hep3Vector hy(y.X(), y.Y(), y.Z());
      CLHEP::Hep3Vector hz(z.X(), z.Y(), z.Z());
      CLHEP::HepRotation hrot(hx, hy, hz);
      CLHEP::Hep3Vector htran ( fv.translation().X(),
				fv.translation().Y(),
				fv.translation().Z()  );

      const HepGeom::Transform3D ht3d ( hrot,  // only scale translation
					CaloCellGeometry::k_ScaleFromDDDtoGeant*htran ) ;    

      const HepGeom::Point3D<float> ctr (
	 ht3d*HepGeom::Point3D<float> (0,0,0) ) ;

      const GlobalPoint refPoint ( ctr.x(), ctr.y(), ctr.z() ) ;

      std::vector<CCGFloat> vv ;
      vv.reserve( pv.size() + 1 ) ;
      for( unsigned int i ( 0 ) ; i != pv.size() ; ++i )
      {
	 vv.push_back( CaloCellGeometry::k_ScaleFromDDDtoGeant*pv[i] ) ;
      }
      vv.push_back( 0. ) ; // tilt=0 here
      const CCGFloat* pP ( CaloCellGeometry::getParmPtr( vv, 
							 ebg->parMgr(), 
							 ebg->parVecVec() ) ) ;

      const DetId detId ( getDetIdForDDDNode(fv) ) ;

      //Adding cell to the Geometry

      ebg->newCell( refPoint, refPoint, refPoint,
		    pP, 
		    detId ) ;
   } // loop over all children
}

unsigned int 
EcalTBHodoscopeGeometryLoaderFromDDD::getDetIdForDDDNode(
   const DDFilteredView &fv )
{
   // perform some consistency checks
   // get the parents and grandparents of this node
   DDGeoHistory parents = fv.geoHistory();
  
   assert(parents.size() >= 3);

   EcalBaseNumber baseNumber;
   //baseNumber.setSize(parents.size());

   for( unsigned int i=1 ;i <= parents.size(); i++)
   {
      baseNumber.addLevel( parents[ parents.size() - i ].logicalPart().name().name(),
			   parents[ parents.size() - i ].copyno() ) ;
   }

   return _scheme.getUnitID( baseNumber ) ;
}

DDFilter* EcalTBHodoscopeGeometryLoaderFromDDD::getDDFilter()
{
   DDSpecificsFilter *filter = new DDSpecificsFilter();

   filter->setCriteria( DDValue( "SensitiveDetector",
				 "EcalTBH4BeamDetector",
				 0 ),
			DDCompOp::equals,
			DDLogOp::AND,
			true,
			true ) ;

   filter->setCriteria( DDValue( "ReadOutName",
				 "EcalTBH4BeamHits",
				 0 ),
			DDCompOp::equals,
			DDLogOp::AND,
			true,
			true ) ;
   return filter;
}
