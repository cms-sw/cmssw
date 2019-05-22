#include "DetectorDescription/Parser/src/DDLCompositeMaterial.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/src/DDLMaterial.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstddef>
#include <map>
#include <utility>

class DDCompactView;

DDLCompositeMaterial::DDLCompositeMaterial(DDLElementRegistry* myreg) : DDLMaterial(myreg) {}

// to initialize the CompositeMaterial, clear all rMaterials in case some other
// rMaterial was used for some other element.
void DDLCompositeMaterial::preProcessElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  // fyi: no need to clear MaterialFraction because it is cleared at the end of each
  // CompositeMaterial
  myRegistry_->getElement("rMaterial")->clear();
}

void DDLCompositeMaterial::processElement(const std::string& name, const std::string& nmspace, DDCompactView& cpv) {
  ClhepEvaluator& ev = myRegistry_->evaluator();
  DDXMLAttribute atts = getAttributeSet();

  DDName ddn = getDDName(nmspace);
  DDMaterial mat;

  mat = DDMaterial(ddn, ev.eval(nmspace, atts.find("density")->second));

  // Get references to relevant DDL elements that are needed.
  auto myMF = myRegistry_->getElement("MaterialFraction");
  auto myrMaterial = myRegistry_->getElement("rMaterial");

  // Get the names from those elements and also the namespace for the reference element.
  // The parent element CompositeMaterial MUST be in the same namespace as this fraction.
  // additionally, because it is NOT a reference, we do not try to dis-entangle the namespace.
  // That is, we do not use the getName() which searches the name for a colon, but instead use
  // the "raw" name attribute.

  // TO DO:  sfractions assumes that the values are "mixture by weight" (I think)
  // we need to retrieve the fraction attributes and then check the method
  // attribute of the CompositeMaterial to determine if the numbers go straight
  // in to the DDCore or if the numbers need to be manipulated so that the
  // sfractions are really mixtures by weight going into the DDCore.  Otherwise,
  // the DDCore has to know which are which and right now it does not.

  if (myMF->size() != myrMaterial->size()) {
    std::string msg = "/nDDLCompositeMaterial::processElement found that the ";
    msg += "number of MaterialFractions does not match the number ";
    msg += "of rMaterial names for ";
    msg += ddn.ns() + ":" + ddn.name() + ".";
    throwError(msg);
  }
  for (size_t i = 0; i < myrMaterial->size(); ++i) {
    atts = myMF->getAttributeSet(i);
    mat.addMaterial(myrMaterial->getDDName(nmspace, "name", i), ev.eval(nmspace, atts.find("fraction")->second));
  }
  // clears and sets new reference to THIS material.
  DDLMaterial::setReference(nmspace, cpv);
  myMF->clear();
  clear();
}
