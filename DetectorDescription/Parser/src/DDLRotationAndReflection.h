#ifndef DDL_RotationAndReflection_H
#define DDL_RotationAndReflection_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

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
  DDLRotationAndReflection( DDLElementRegistry* myreg );

  /// Destructor
  ~DDLRotationAndReflection( void );

  /// returns 1 = left handed rotation matrix, 0 = right-handed, -1 = not orthonormal.
  int isLeftHanded( DD3Vector x, DD3Vector y, DD3Vector z, const std::string & nmspace );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

private:

  DD3Vector makeX( std::string nmspace );
  DD3Vector makeY( std::string nmspace );
  DD3Vector makeZ( std::string nmspace );
};
#endif
