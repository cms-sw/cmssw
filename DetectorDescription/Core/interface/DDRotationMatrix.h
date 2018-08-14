#ifndef DDD_DDRotationMatrix_h
#define DDD_DDRotationMatrix_h

#include <Math/Rotation3D.h>
#include <Math/AxisAngle.h>

//! A DDRotationMatrix is currently implemented with a ROOT Rotation3D
using DDRotationMatrix = ROOT::Math::Rotation3D;
using DDAxisAngle = ROOT::Math::AxisAngle;

#endif
