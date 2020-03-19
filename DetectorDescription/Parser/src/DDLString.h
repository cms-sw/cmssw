#ifndef DDL_String_H
#define DDL_String_H

#include <map>
#include <string>
#include <vector>

#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDString.h"

class DDCompactView;
class DDLElementRegistry;

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
class DDLString final : public DDXMLElement {
public:
  DDLString(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
};

#endif
