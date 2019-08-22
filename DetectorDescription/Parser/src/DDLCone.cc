#include "DetectorDescription/Parser/src/DDLCone.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

class DDCompactView;

DDLCone::DDLCone(DDLElementRegistry* myreg) : DDLSolid(myreg) {}

void DDLCone::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  ClhepEvaluator& ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDSolid ddcone = DDSolidFactory::cons(getDDName(nmspace),
                                        ev.eval(nmspace, atts.find("dz")->second),
                                        ev.eval(nmspace, atts.find("rMin1")->second),
                                        ev.eval(nmspace, atts.find("rMax1")->second),
                                        ev.eval(nmspace, atts.find("rMin2")->second),
                                        ev.eval(nmspace, atts.find("rMax2")->second),
                                        ev.eval(nmspace, atts.find("startPhi")->second),
                                        ev.eval(nmspace, atts.find("deltaPhi")->second));

  DDLSolid::setReference(nmspace, cpv);
}
