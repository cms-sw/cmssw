#include "DetectorDescription/Parser/src/DDLShapelessSolid.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLShapelessSolid::DDLShapelessSolid(DDLElementRegistry* myreg) : DDLSolid(myreg) {}

void DDLShapelessSolid::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  myRegistry_->getElement("rSolid")->clear();
}

// Upon ending a ShapelessSolid element, call DDCore giving the box name, and dimensions.
void DDLShapelessSolid::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  DDSolid dds = DDSolidFactory::shapeless(getDDName(nmspace));

  DDLSolid::setReference(nmspace, cpv);
}
