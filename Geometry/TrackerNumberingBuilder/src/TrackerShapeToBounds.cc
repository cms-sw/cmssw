#include "Geometry/TrackerNumberingBuilder/interface/TrackerShapeToBounds.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <algorithm>
#include <iostream>
//#define DEBUG

/* find out about the rotations of the detectors:
       
  (the code should also find out about other detector-types (pixes-fw, ...)	  	
  currently not implemented, of course)

  - for pixel-barrels:
  detectors are modelled by boxes, ORCA convention for the local frame:
  . the thickness is in global r-direction of global-CMS
  . the longer side if in z-direction of global-CMS
  . the shorter side is in phi-direction of global-CMS
  ORCA convention of the local-frame:
  . the local z-axis is defined to be in direction of the thickness of the box
  . the local y-axis is defined to be in direction of the longer side of the box
  . the local x-axis is thus in direction of the shorter side of the box
	
  1. So first look how the detector box is defined in DDD (which axis direction
  is the thickness, which axis-direction is the shorter side,...)
  2. Define a rotation which reorientates the box to Orca-conventions
  in the local frame, if necessary
  3. combine the global rotation from DDD with the rotation defined in 2.   
  */

Bounds* TrackerShapeToBounds::buildBounds(const cms::DDSolidShape& shape, const std::vector<double>& par) const {
  switch (shape) {
    case cms::DDSolidShape::ddbox:
      return buildBox(par);
      break;
    case cms::DDSolidShape::ddtrap:
      return buildTrap(par);
      break;
    case cms::DDSolidShape::ddtubs:
    case cms::DDSolidShape::ddpolycone:
    case cms::DDSolidShape::ddsubtraction:
      return buildOpen();
      break;
    default:
      std::cout << "Wrong DDshape to build...." << cms::dd::name(cms::DDSolidShapeMap, shape) << std::endl;
      Bounds* bounds = nullptr;
      return bounds;
  }
}

Bounds* TrackerShapeToBounds::buildBox(const std::vector<double>& paras) const {
  int indexX = 0;
  int indexY = 1;
  int indexZ = 2;
  Bounds* bounds = nullptr;

  if (paras[1] < paras[0] && paras[0] < paras[2]) {
    indexX = 0;
    indexY = 2;
    indexZ = 1;
  }

  bounds = new RectangularPlaneBounds(paras[indexX] / cm,   // width - shorter side
                                      paras[indexY] / cm,   // length - longer side
                                      paras[indexZ] / cm);  // thickness
  return bounds;
}

Bounds* TrackerShapeToBounds::buildTrap(const std::vector<double>& paras) const {
  Bounds* bounds = nullptr;
  /*
    TrapezoidalPlaneBounds (float be, float te, float a, float t)
    constructed from:
    half bottom edge (smaller side width)
    half top edge (larger side width)
    half apothem (distance from top to bottom sides, measured perpendicularly to them)
    half thickness.
    
    if  we have indexX=0, indexY=1 and indeZ=2
    4 = be (ORCA x)
    9 = te (ORCA x)
    0 = a (ORCA y)
    3 = t (ORCA z)

    if  we have indexX=0, indexY=2 and indeZ=1
    4 = be (ORCA x)
    9 = te (ORCA x)
    3 = a (ORCA y)
    0 = t (ORCA z)

    so, so we have the indexes:
    if indexX==0, indexY==1, indexZ==2, then everything is ok and
    the following orcaCorrection-rotation will be a unit-matrix.
  */

  if (paras[0] < 5) {
    bounds = new TrapezoidalPlaneBounds(paras[4] / cm, paras[9] / cm, paras[3] / cm, paras[0] / cm);
  } else if (paras[0] > paras[3]) {
    bounds = new TrapezoidalPlaneBounds(paras[4] / cm, paras[9] / cm, paras[0] / cm, paras[3] / cm);
  }
  return bounds;
}

Bounds* TrackerShapeToBounds::buildOpen() const {
  OpenBounds* bounds = new OpenBounds();
  return bounds;
}
