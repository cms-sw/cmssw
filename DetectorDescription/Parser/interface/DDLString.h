#ifndef DDL_String_H
#define DDL_String_H

// -------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------
#include "DetectorDescription/DDParser/interface/DDXMLElement.h"
#include "DetectorDescription/DDCore/interface/DDString.h"
#include "DetectorDescription/DDBase/interface/DDTypes.h"

#include <string>
#include <vector>
#include <map>

///  DDLString handles String Elements.
/** @class DDLString
 * @author Michael Case
 *
 *  DDLString.h  -  description
 *  -------------------
 *  begin: Fri Nov 21 2003
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 */
class DDLString : public DDXMLElement
{

 public:

  DDLString();

  ~DDLString();

  void preProcessElement (const std::string& name, const std::string& nmspace);

  void processElement (const std::string& name, const std::string& nmspace);

 private:

};
#endif
