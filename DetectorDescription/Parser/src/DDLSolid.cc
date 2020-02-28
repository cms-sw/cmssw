#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <map>
#include <utility>
#include <vector>

class DDCompactView;

DDLSolid::DDLSolid(DDLElementRegistry* myreg) : DDXMLElement(myreg) {}

void DDLSolid::setReference(const std::string& nmspace, DDCompactView& cpv) {
  // in case it was a BooleanSolid or a ReflectionSolid, clear rSolid.
  auto myrSolid = myRegistry_->getElement("rSolid");
  myrSolid->clear();

  // Make sure Solid elements are in LogicalPart elements.
  if (parent() == "LogicalPart") {
    auto refsol = myRegistry_->getElement("rSolid");
    std::vector<std::string> names;
    std::vector<std::string> values;
    names.emplace_back("name");
    values.emplace_back(getAttributeSet().find("name")->second);
    refsol->loadAttributes("rSolid", names, values, nmspace, cpv);
  }

  // clear THIS solid's values.
  clear();
}
