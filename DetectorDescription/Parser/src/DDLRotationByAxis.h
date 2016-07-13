#ifndef DDL_RotationByAxis_H
#define DDL_RotationByAxis_H

#include <string>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"
// Base dependency
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"

class DDCompactView;
class DDLElementRegistry;

///  DDLRotationByAxis handles RotationByAxis elements
/** @class DDLRotationByAxis
 * @author Michael Case
 *
 *  DDLRotationByAxis.h  -  description
 *  -------------------
 *  begin: Wed. Nov. 19, 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 *  This is the RotationByAxis element which rotates around an axis.
 *
 */
class DDLRotationByAxis : public DDXMLElement
{
public:

  /// Constructor 
  DDLRotationByAxis( DDLElementRegistry* myreg );

  /// Destructor
  virtual ~DDLRotationByAxis( void );

  virtual void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  virtual void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  virtual DDRotationMatrix processOne( DDRotationMatrix R, std::string& axis, std::string& angle ); 

private:
  std::string pNameSpace;
  std::string pName;
};

#endif
