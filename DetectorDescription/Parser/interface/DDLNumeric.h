#ifndef DDL_Numeric_H
#define DDL_Numeric_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDXMLElement.h"
#include "DetectorDescription/DDCore/interface/DDNumeric.h"
#include "DetectorDescription/DDBase/interface/DDTypes.h"

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

  DDLNumeric();

  ~DDLNumeric();

  void preProcessElement (const std::string& name, const std::string& nmspace);

  void processElement (const std::string& name, const std::string& nmspace);

 private:

};
#endif
