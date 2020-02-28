#include "DetectorDescription/Parser/src/DDLPgonGenerator.h"
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

DDLPgonGenerator::DDLPgonGenerator(DDLElementRegistry* myreg) : DDLSolid(myreg) {}

void DDLPgonGenerator::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  myRegistry_->getElement("XYPoint")->clear();
  myRegistry_->getElement("ZXYSection")->clear();
}

// Upon encountering an end Extruded Polygone tag, process the XYPoints
// element and extract the x and y std::vectors to feed into DDCore. Then, clear
// the XYPoints and clear this element.
void DDLPgonGenerator::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  auto myXYPoints = myRegistry_->getElement("XYPoint");
  auto myZXYSection = myRegistry_->getElement("ZXYSection");

  ClhepEvaluator& ev = myRegistry_->evaluator();
  DDXMLAttribute atts;

  // get x and y
  std::vector<double> x, y;
  for (size_t i = 0; i < myXYPoints->size(); ++i) {
    atts = myXYPoints->getAttributeSet(i);
    auto xit = atts.find("x");
    if (xit != atts.end())
      x.emplace_back(ev.eval(nmspace, xit->second));
    auto yit = atts.find("y");
    if (yit != atts.end())
      y.emplace_back(ev.eval(nmspace, yit->second));
  }
  assert(x.size() == y.size());

  // get zSection information
  std::vector<double> z, zx, zy, zscale;

  for (size_t i = 0; i < myZXYSection->size(); ++i) {
    atts = myZXYSection->getAttributeSet(i);
    auto zit = atts.find("z");
    if (zit != atts.end())
      z.emplace_back(ev.eval(nmspace, zit->second));
    auto xit = atts.find("x");
    if (xit != atts.end())
      zx.emplace_back(ev.eval(nmspace, xit->second));
    auto yit = atts.find("y");
    if (yit != atts.end())
      zy.emplace_back(ev.eval(nmspace, yit->second));
    auto sit = atts.find("scale");
    if (sit != atts.end())
      zscale.emplace_back(std::stod(sit->second));
  }
  assert(z.size() == zx.size());
  assert(z.size() == zy.size());
  assert(z.size() == zscale.size());

  atts = getAttributeSet();
  if (name == "ExtrudedPolygon") {
    DDSolid extrudedpolygon = DDSolidFactory::extrudedpolygon(getDDName(nmspace), x, y, z, zx, zy, zscale);
  } else {
    std::string msg = "\nDDLPgonGenerator::processElement was called with incorrect name of solid: " + name;
    throwError(msg);
  }
  DDLSolid::setReference(nmspace, cpv);

  // clear out XYPoint element accumulator and ZXYSections
  myXYPoints->clear();
  myZXYSection->clear();
  clear();
}
