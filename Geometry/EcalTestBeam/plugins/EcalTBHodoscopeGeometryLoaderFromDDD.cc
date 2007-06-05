#include "Geometry/EcalTestBeam/plugins/EcalTBHodoscopeGeometryLoaderFromDDD.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"


#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"

#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

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
       
       vector<double> parameters = solid.parameters();      

      // use preshower strip as box in space representation
      PreshowerStrip* theGeometry = new PreshowerStrip(parameters[0], parameters[1], parameters[2]);

       // rotate the box and then move it
      DD3Vector x, y, z;
      fv.rotation().GetComponents(x,y,z);
      Hep3Vector hx(x.X(), x.Y(), x.Z());
      Hep3Vector hy(y.X(), y.Y(), y.Z());
      Hep3Vector hz(z.X(), z.Y(), z.Z());
      HepRotation hrot(hx, hy, hz);
      Hep3Vector htran(fv.translation().X(), fv.translation().Y(), fv.translation().Z());
      theGeometry->hepTransform(HepScale3D(0.1) * // convert from mm (DDD) to cm (G3)
				HepTransform3D(hrot,htran));
      
      //Adding cell to the Geometry
      ebg->addCell(DetId(getDetIdForDDDNode(fv)),theGeometry);
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
