#ifndef DDL_Numeric_H
#define DDL_Numeric_H

#include <map>
#include <string>
#include <vector>

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDNumeric.h"

class DDCompactView;
class DDLElementRegistry;

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
class DDLNumeric final : public DDXMLElement
{
public:

  DDLNumeric( DDLElementRegistry* myreg );

  void preProcessElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
  void processElement( const std::string& name, const std::string& nmspace, DDCompactView& cpv ) override;
};
#endif
