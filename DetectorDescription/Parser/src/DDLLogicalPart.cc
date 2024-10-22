#include "DetectorDescription/Parser/src/DDLLogicalPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <utility>

class DDCompactView;

DDLLogicalPart::DDLLogicalPart(DDLElementRegistry* myreg) : DDXMLElement(myreg) {
  // initialize category map
  catMap_["sensitive"] = DDEnums::sensitive;
  catMap_["cable"] = DDEnums::cable;
  catMap_["cooling"] = DDEnums::cooling;
  catMap_["support"] = DDEnums::support;
  catMap_["envelope"] = DDEnums::envelope;
  catMap_["unspecified"] = DDEnums::unspecified;
}

// upon initialization, we want to clear rMaterial and rSolid.
void DDLLogicalPart::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  myRegistry_->getElement("rMaterial")->clear();
  myRegistry_->getElement("rSolid")->clear();
}

// Upon encountering the end of the LogicalPart element, retrieve all
// relevant information from its sub-elements and put it all together to
// call the DDCore appropriately.
//
// History:  Initially only rSolid and rMaterial elements worked.  Now,
// a Material or a Solid element inside the LogicalPart also works but
// this is handled outside this class.  Each Solid inherits from DDLSolid
// and in each Solid, the processElement method must call the setReference
// of the DDLSolid.  The Material element also works similarly.  Thus,
// by retrieving the rMaterial and the rSolid it actually will be handling
// Material and Solid subelements as well.

void DDLLogicalPart::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  // rMaterial and rSolid
  auto myrMaterial = myRegistry_->getElement("rMaterial");  // get Material reference child
  auto myrSolid = myRegistry_->getElement("rSolid");        // get Solid reference child

  DDXMLAttribute atts = getAttributeSet();

  DDSolid mySolid = DDSolid(myrSolid->getDDName(nmspace));
  DDMaterial myMaterial = DDMaterial(myrMaterial->getDDName(nmspace));

  DDEnums::Category cat;

  if (atts.find("category") != atts.end())
    cat = catMap_[atts.find("category")->second];
  else
    cat = catMap_["unspecified"];

  DDLogicalPart lp(getDDName(nmspace), myMaterial, mySolid, cat);

  // clear all "children" and attributes
  myrMaterial->clear();
  myrSolid->clear();

  // after each logical part is made, we can clear it
  clear();
}
