#ifndef DetectorDescription_DDCMS_DDTranslation_h
#define DetectorDescription_DDCMS_DDTranslation_h
#include <Math/Vector3D.h>

//! A DD Translation is currently implemented with Root Vector3D
typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > DD3Vector;
typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > DDTranslation;

#endif
