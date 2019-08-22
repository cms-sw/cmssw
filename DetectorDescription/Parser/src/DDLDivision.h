#ifndef DDL_Division_H
#define DDL_Division_H

#include <map>
#include <string>

#include "DDDividedGeometryObject.h"
#include "DDXMLElement.h"
#include "DetectorDescription/Core/interface/DDDivision.h"

class DDCompactView;
class DDDividedGeometryObject;
class DDLElementRegistry;

/// DDLDivision processes Division elements.
/** @class DDLDivision
 * @author Michael Case
 *
 *  DDLDivision.h  -  description
 *  -------------------
 *  begin: Friday, April 23, 2004
 *  email: case@ucdhep.ucdavis.edu
 *
 *
 */

class DDLDivision final : public DDXMLElement {
public:
  DDLDivision(DDLElementRegistry* myreg);

  void preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;
  void processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) override;

private:
  DDDividedGeometryObject* makeDivider(const DDDivision& div, DDCompactView* cpv);
};

#endif
