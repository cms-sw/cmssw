#include "DetectorDescription/Parser/src/DDLPolyGenerator.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLSolid.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

class DDCompactView;

DDLPolyGenerator::DDLPolyGenerator(DDLElementRegistry* myreg) : DDLSolid(myreg) {}

void DDLPolyGenerator::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  myRegistry_->getElement("RZPoint")->clear();
  myRegistry_->getElement("ZSection")->clear();
}

// Upon encountering an end Polycone or Polyhedra tag, process the RZPoints
// element and extract the r and z std::vectors to feed into DDCore.  Then, clear
// the RZPoints and clear this element.
void DDLPolyGenerator::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  auto myRZPoints = myRegistry_->getElement("RZPoint");
  auto myZSection = myRegistry_->getElement("ZSection");

  ClhepEvaluator& ev = myRegistry_->evaluator();
  DDXMLAttribute atts;

  // get z and r
  std::vector<double> z, r, x, y;
  for (size_t i = 0; i < myRZPoints->size(); ++i) {
    atts = myRZPoints->getAttributeSet(i);
    z.emplace_back(ev.eval(nmspace, atts.find("z")->second));
    r.emplace_back(ev.eval(nmspace, atts.find("r")->second));
  }

  // if z is empty, then it better not have been a polycone defined
  // by RZPoints, instead, it must be a ZSection defined polycone.
  if (z.empty()) {
    // get zSection information, note, we already have a z declared above
    // and we will use r for rmin.  In this case, no use "trying" because
    // it better be there!
    std::vector<double> rMax;

    for (size_t i = 0; i < myZSection->size(); ++i) {
      atts = myZSection->getAttributeSet(i);
      z.emplace_back(ev.eval(nmspace, atts.find("z")->second));
      r.emplace_back(ev.eval(nmspace, atts.find("rMin")->second));
      rMax.emplace_back(ev.eval(nmspace, atts.find("rMax")->second));
    }
    atts = getAttributeSet();
    if (name == "Polycone")  // defined with ZSections
    {
      DDSolid ddpolycone = DDSolidFactory::polycone(getDDName(nmspace),
                                                    ev.eval(nmspace, atts.find("startPhi")->second),
                                                    ev.eval(nmspace, atts.find("deltaPhi")->second),
                                                    z,
                                                    r,
                                                    rMax);
    } else if (name == "Polyhedra")  // defined with ZSections
    {
      DDSolid ddpolyhedra = DDSolidFactory::polyhedra(getDDName(nmspace),
                                                      int(ev.eval(nmspace, atts.find("numSide")->second)),
                                                      ev.eval(nmspace, atts.find("startPhi")->second),
                                                      ev.eval(nmspace, atts.find("deltaPhi")->second),
                                                      z,
                                                      r,
                                                      rMax);
    }

  } else if (name == "Polycone")  // defined with RZPoints
  {
    atts = getAttributeSet();
    DDSolid ddpolycone = DDSolidFactory::polycone(getDDName(nmspace),
                                                  ev.eval(nmspace, atts.find("startPhi")->second),
                                                  ev.eval(nmspace, atts.find("deltaPhi")->second),
                                                  z,
                                                  r);
  } else if (name == "Polyhedra")  // defined with RZPoints
  {
    atts = getAttributeSet();
    DDSolid ddpolyhedra = DDSolidFactory::polyhedra(getDDName(nmspace),
                                                    int(ev.eval(nmspace, atts.find("numSide")->second)),
                                                    ev.eval(nmspace, atts.find("startPhi")->second),
                                                    ev.eval(nmspace, atts.find("deltaPhi")->second),
                                                    z,
                                                    r);
  } else {
    std::string msg = "\nDDLPolyGenerator::processElement was called with incorrect name of solid: " + name;
    throwError(msg);
  }
  DDLSolid::setReference(nmspace, cpv);

  // clear out RZPoint element accumulator and ZSections
  myRZPoints->clear();
  myZSection->clear();
  clear();
}
