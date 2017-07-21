#ifndef DDL_RotationAndReflection_H
#define DDL_RotationAndReflection_H

#include <string>

#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"

class DDCompactView;
class DDLElementRegistry;

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
class DDLRotationAndReflection final : public DDXMLElement
{
 public:

  DDLRotationAndReflection( DDLElementRegistry* myreg );

  /// returns 1 = left handed rotation matrix, 0 = right-handed, -1 = not orthonormal.
  int isLeftHanded( DD3Vector x, DD3Vector y, DD3Vector z, const std::string & nmspace );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;

private:

  DD3Vector makeX( std::string nmspace );
  DD3Vector makeY( std::string nmspace );
  DD3Vector makeZ( std::string nmspace );
};

#endif
