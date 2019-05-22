#include "DetectorDescription/Parser/src/DDLElementaryMaterial.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLMaterial.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <map>
#include <utility>

class DDCompactView;

DDLElementaryMaterial::DDLElementaryMaterial(DDLElementRegistry* myreg) : DDLMaterial(myreg) {}

void DDLElementaryMaterial::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  ClhepEvaluator& ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDMaterial mat = DDMaterial(getDDName(nmspace),
                              ev.eval(nmspace, atts.find("atomicNumber")->second),
                              ev.eval(nmspace, atts.find("atomicWeight")->second),
                              ev.eval(nmspace, atts.find("density")->second));

  DDLMaterial::setReference(nmspace, cpv);
  clear();
}
