///  \author    : Gero Flucke
///  date       : October 2010
///  $Revision$
///  $Date$
///  (last update by $Author$)

#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformationFactory.h"
#include "Geometry/CommonTopologies/interface/BowedSurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/TwoBowedSurfacesDeformation.h"

// included by header:
// #include <vector>

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
