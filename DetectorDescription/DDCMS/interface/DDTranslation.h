#ifndef DetectorDescription_DDCMS_DDTranslation_h
#define DetectorDescription_DDCMS_DDTranslation_h

#include <Math/Vector3D.h>

//! A DD Translation is currently implemented with Root Vector3D
using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
using DDTranslation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

#endif
