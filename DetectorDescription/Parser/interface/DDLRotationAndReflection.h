#ifndef DDL_RotationAndReflection_H
#define DDL_RotationAndReflection_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDXMLElement.h"

// CLHEP dependencies
#include "CLHEP/Geometry/Transform3D.h"

#include <string>

///  DDLRotationAndReflection handles RotationCMSIM and ReflectionRotation elements.
/** @class DDLRotationAndReflection
 * @author Michael Case
 *
 *  DDLRotationAndReflection.h  -  description
 *  -------------------
 *  begin: Tue Oct 30 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 *  This is the Rotation and Reflection element processor.
 *
 */
class DDLRotationAndReflection : public DDXMLElement
{

 public:

  /// Constructor 
  DDLRotationAndReflection();

  /// Destructor
  ~DDLRotationAndReflection();

  /// returns 1 = left handed rotation matrix, 0 = right-handed, -1 = not orthonormal.
  int isLeftHanded(Hep3Vector x, Hep3Vector y, Hep3Vector z, const string & nmspace);

  void processElement (const std::string& name, const std::string& nmspace);

 private:

  Hep3Vector makeX(std::string nmspace);
  Hep3Vector makeY(std::string nmspace);
  Hep3Vector makeZ(std::string nmspace);
};
#endif
