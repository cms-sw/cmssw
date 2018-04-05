///  \author    : Gero Flucke
///  date       : October 2010

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
    return kNoDeformations; // not reached, to please the compiler
  }
}
std::string SurfaceDeformationFactory::surfaceDeformationTypeName(const SurfaceDeformationFactory::Type &type){
  switch (type){
  case kBowedSurface:
    return std::string("BowedSurface");
  case kTwoBowedSurfaces:
    return std::string("TwoBowedSurfaces");
  default:
    throw cms::Exception("BadInput") << "SurfaceDeformationFactory::surfaceDeformationTypeName: "
      << "Unknown SurfaceDeformation type " << type
      << " (must be 'kBowedSurface' or 'kTwoBowedSurfaces'.\n";
    return std::string("NoDeformations");
  }
}


SurfaceDeformation* SurfaceDeformationFactory::create(int type, const std::vector<double> &params)
{
  switch(type){
  case kBowedSurface:
  case kTwoBowedSurfaces:
    return SurfaceDeformationFactory::create(params);
  default:
    throw cms::Exception("BadInput") << "SurfaceDeformationFactory::create: "
      << "Unknown SurfaceDeformation type " << type << " (need "
      << kBowedSurface << " or " << kTwoBowedSurfaces
      << ")\n";
    return nullptr;
  }
}

SurfaceDeformation* SurfaceDeformationFactory::create(const std::vector<double> &params)
{
      if (params.size() <= BowedSurfaceDeformation::maxParameterSize() &&
	  params.size() >= BowedSurfaceDeformation::minParameterSize()) 
	return new BowedSurfaceDeformation(params);
      else if (params.size() <= TwoBowedSurfacesDeformation::maxParameterSize() &&
	  params.size() >= TwoBowedSurfacesDeformation::minParameterSize())
	return new TwoBowedSurfacesDeformation(params);

  throw cms::Exception("BadInput") << "SurfaceDeformationFactory::create: "
				   << "Params.size() (" << params.size()
				   << ") does not match.\n";
  
  return nullptr;
}
