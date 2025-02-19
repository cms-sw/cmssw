///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision: 1.2 $
///  $Date: 2011/02/11 10:57:40 $
///  (last update by $Author: flucke $)

#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"
#include "Geometry/CommonTopologies/interface/BowedSurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/TwoBowedSurfacesDeformation.h"

// included by header:
// #include <vector>
// #include <string>

SurfaceDeformationFactory::Type
SurfaceDeformationFactory::surfaceDeformationType(const std::string &typeString)
{
  if      (typeString == "BowedSurface")     return kBowedSurface;
  else if (typeString == "TwoBowedSurfaces") return kTwoBowedSurfaces;
  else {
    throw cms::Exception("BadInput") << "SurfaceDeformationFactory::surfaceDeformationType: "
				     << "Unknown SurfaceDeformation type " << typeString
				     << " (must be 'BowedSurface' or 'TwoBowedSurfaces'.\n";
    return kBowedSurface; // not reached, to please the compiler
  }
}

SurfaceDeformation* SurfaceDeformationFactory::create(int type, const std::vector<double> &params)
{
  switch(type) {
  case kBowedSurface:
    {
      if (params.size() <= BowedSurfaceDeformation::maxParameterSize() &&
	  params.size() >= BowedSurfaceDeformation::minParameterSize()) {
	return new BowedSurfaceDeformation(params);
      } else {
	break;
      }
    }
  case kTwoBowedSurfaces:
    {
      if (params.size() <= TwoBowedSurfacesDeformation::maxParameterSize() &&
	  params.size() >= TwoBowedSurfacesDeformation::minParameterSize()) {
	return new TwoBowedSurfacesDeformation(params);
      } else {
	break;
      }
    }
  }

  throw cms::Exception("BadInput") << "SurfaceDeformationFactory::create: "
				   << "Unknown SurfaceDeformation type " << type << " (need "
				   << kBowedSurface << " or " << kTwoBowedSurfaces 
				   << ") or params.size() (" << params.size()
				   << ") does not match.\n";
  
  return 0;
}
