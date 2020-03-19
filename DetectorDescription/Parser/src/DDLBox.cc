#include "DetectorDescription/Parser/src/DDLBox.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"

#include <map>
#include <utility>

class DDCompactView;

DDLBox::DDLBox(DDLElementRegistry* myreg) : DDLSolid(myreg) {}

// Upon ending a Box element, call DDCore giving the box name, and dimensions.
void DDLBox::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  ClhepEvaluator& ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDName ddname = getDDName(nmspace);
  DDSolid ddbox = DDSolidFactory::box(ddname,
                                      ev.eval(nmspace, atts.find("dx")->second),
                                      ev.eval(nmspace, atts.find("dy")->second),
                                      ev.eval(nmspace, atts.find("dz")->second));
  // Attempt to make sure Solid elements can be in LogicalPart elements.
  DDLSolid::setReference(nmspace, cpv);
}
