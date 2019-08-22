#include "DetectorDescription/Parser/src/DDLAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmHandler.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLMap.h"
#include "DetectorDescription/Parser/src/DDLVector.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include <cstddef>
#include <map>
#include <utility>

class DDCompactView;

DDLAlgorithm::DDLAlgorithm(DDLElementRegistry* myreg) : DDXMLElement(myreg) {}

void DDLAlgorithm::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  myRegistry_->getElement("Vector")->clear();
}

void DDLAlgorithm::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  auto myNumeric = myRegistry_->getElement("Numeric");
  auto myString = myRegistry_->getElement("String");
  auto myVector = myRegistry_->getElement("Vector");
  auto myMap = myRegistry_->getElement("Map");
  auto myrParent = myRegistry_->getElement("rParent");

  DDName algoName(getDDName(nmspace));
  DDLogicalPart lp(DDName(myrParent->getDDName(nmspace)));
  DDXMLAttribute atts;

  // handle all Numeric elements in the Algorithm.
  DDNumericArguments nArgs;
  size_t i = 0;
  for (; i < myNumeric->size(); ++i) {
    atts = myNumeric->getAttributeSet(i);
    nArgs[atts.find("name")->second] = myRegistry_->evaluator().eval(nmspace, atts.find("value")->second);
  }

  DDStringArguments sArgs;
  for (i = 0; i < myString->size(); ++i) {
    atts = myString->getAttributeSet(i);
    sArgs[atts.find("name")->second] = atts.find("value")->second;
  }

  DDAlgorithmHandler handler;
  atts = getAttributeSet();
  handler.initialize(algoName,
                     lp,
                     nArgs,
                     static_cast<DDLVector*>(myVector.get())->getMapOfVectors(),
                     static_cast<DDLMap*>(myMap.get())->getMapOfMaps(),
                     sArgs,
                     static_cast<DDLVector*>(myVector.get())->getMapOfStrVectors());
  handler.execute(cpv);

  // clear used/referred to elements.
  myString->clear();
  myNumeric->clear();
  myVector->clear();
  myMap->clear();
  myrParent->clear();
  clear();
}
