#ifndef DDL_Numeric_H
#define DDL_Numeric_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Base/interface/DDTypes.h"

#include <string>
#include <vector>
#include <map>

///  DDLNumeric handles Numeric Elements
/** @class DDLNumeric
 * @author Michael Case
 *
 *  DDLNumeric.h  -  description
 *  -------------------
 *  begin: Fri Nov 21 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 */
class DDLNumeric : public DDXMLElement
{
public:

  DDLNumeric( DDLElementRegistry* myreg );

  ~DDLNumeric( void );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );

  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv );
};
#endif
