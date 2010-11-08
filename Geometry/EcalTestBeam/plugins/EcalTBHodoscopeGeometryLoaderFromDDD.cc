#include "Geometry/EcalTestBeam/plugins/EcalTBHodoscopeGeometryLoaderFromDDD.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"


#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"


#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDName.h"
//#include "DetectorDescription/Core/interface/DDInit.h"


#include <iostream>
#include <vector>

EcalTBHodoscopeGeometryLoaderFromDDD::EcalTBHodoscopeGeometryLoaderFromDDD(): _scheme(0) 
{
  _scheme=new EcalHodoscopeNumberingScheme();
}

std::auto_ptr<CaloSubdetectorGeometry> EcalTBHodoscopeGeometryLoaderFromDDD::load(const DDCompactView* cpv) {
  std::cout << "[EcalTBHodoscopeGeometryLoaderFromDDD]:: start the construction of EcalTBHodoscope" << std::endl;
  std::auto_ptr<CaloSubdetectorGeometry> ebg(new CaloSubdetectorGeometry());
  makeGeometry(cpv,dynamic_cast<CaloSubdetectorGeometry*>(ebg.get()));
  std::cout << "[EcalTBHodoscopeGeometryLoaderFromDDD]:: Returning EcalTBHodoscopeGeometry" << std::endl;
  return ebg;
}

void EcalTBHodoscopeGeometryLoaderFromDDD::makeGeometry(const DDCompactView* cpv,CaloSubdetectorGeometry* ebg)
{

   if( ebg->cornersMgr() == 0 ) ebg->allocateCorners( 256 ) ;
   if( ebg->parMgr()     == 0 ) ebg->allocatePar( 10, 3 ) ;
  
  DDFilter* filter = getDDFilter();

  DDFilteredView fv(*cpv);
  fv.addFilter(*filter);
  
  bool doSubDets;
  for (doSubDets = fv.firstChild(); doSubDets ; doSubDets = fv.nextSibling())
    {
      
#if 0
      string answer = getDDDString("ReadOutName",&fv);
      if (answer != "EcalTBH4BeamHits")
        continue;
#endif
      
      const DDSolid & solid = fv.logicalPart().solid();

      if (solid.shape() != ddbox) {
        throw DDException(std::string(__FILE__) 
                          +"\n CaloGeometryEcalTBHodoscope::upDate(...): currently only box fiber shapes supported ");
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
      CLHEP::Hep3Vector htran(fv.translation().X(), fv.translation().Y(), fv.translation().Z());

      const HepGeom::Transform3D ht3d ( hrot,                        // only scale translation
				  CaloCellGeometry::k_ScaleFromDDDtoGeant*htran ) ;    


      const HepGeom::Point3D<double>  ctr ( ht3d*HepGeom::Point3D<double> (0,0,0) ) ;

      const GlobalPoint refPoint ( ctr.x(), ctr.y(), ctr.z() ) ;


      std::vector<double> vv ;
      vv.reserve( pv.size() ) ;
      for( unsigned int i ( 0 ) ; i != pv.size() ; ++i )
      {
	 vv.push_back( CaloCellGeometry::k_ScaleFromDDDtoGeant*pv[i] ) ;
      }
      const double* pP ( CaloCellGeometry::getParmPtr( vv, 
						       ebg->parMgr(), 
						       ebg->parVecVec() ) ) ;

      PreshowerStrip* cell ( new PreshowerStrip( refPoint,
						 ebg->cornersMgr(),
						 pP ) ) ;

      //Adding cell to the Geometry
      ebg->addCell(DetId(getDetIdForDDDNode(fv)),cell);
    } // loop over all children (i.e. crystals)
}

unsigned int EcalTBHodoscopeGeometryLoaderFromDDD::getDetIdForDDDNode(const DDFilteredView &fv)
{
  // perform some consistency checks
  // get the parents and grandparents of this node
  DDGeoHistory parents = fv.geoHistory();
  
  assert(parents.size() >= 3);

  EcalBaseNumber baseNumber;
  //baseNumber.setSize(parents.size());

  for (unsigned int i=1 ;i <= parents.size(); i++)
    baseNumber.addLevel(parents[parents.size()-i].logicalPart().name().name(),parents[parents.size()-i].copyno());

  return (_scheme->getUnitID(baseNumber));
  
}

DDFilter* EcalTBHodoscopeGeometryLoaderFromDDD::getDDFilter()
{
  DDSpecificsFilter *filter = new DDSpecificsFilter();
  filter->setCriteria(DDValue("SensitiveDetector","EcalTBH4BeamDetector",0),DDSpecificsFilter::equals,DDSpecificsFilter::AND,true,true);
  filter->setCriteria(DDValue("ReadOutName","EcalTBH4BeamHits",0),DDSpecificsFilter::equals,DDSpecificsFilter::AND,true,true);
  return filter;
}
