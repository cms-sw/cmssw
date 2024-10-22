#ifndef DDL_LogicalPart_H
#define DDL_LogicalPart_H

#include <map>
#include <string>

#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDEnums.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

class DDCompactView;
class DDLElementRegistry;

/// DDLLogicalPart processes LogicalPart elements.
/** @class DDLLogicalPart
 * @author Michael Case
 *
 *  DDLLogicalPart.h  -  description
 *  -------------------
 *  begin: Tue Oct 31 2001
 *  email: case@ucdhep.ucdavis.edu
 *
 *  LogicalPart elements simply have the name attribute.  However, they
 *  also contain elements rSolid and rMaterial.  These come together in
 *  implementation, but as an XML element the only relevant information to
 *  the DDCore is the name attribute.   Optionally, they can have instead
 *  any Solid or Material elements.  To handle the fact that those
 *  elements must "stand alone" and also work within LogicalParts, each 
 *  of those must create a reference (rMaterial or rSolid) to the most
 *  recently processed such element.
 *
 */

class DDLLogicalPart final : public DDXMLElement {
public:
  DDLLogicalPart(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;

private:
  std::map<std::string, DDEnums::Category> catMap_;
};

#endif
