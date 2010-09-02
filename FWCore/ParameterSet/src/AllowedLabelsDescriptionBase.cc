
#include "FWCore/ParameterSet/interface/AllowedLabelsDescriptionBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include "boost/bind.hpp"

#include <iomanip>
#include <ostream>

namespace edm {

  AllowedLabelsDescriptionBase::~AllowedLabelsDescriptionBase() { }

  AllowedLabelsDescriptionBase::
  AllowedLabelsDescriptionBase(std::string const& label, ParameterTypes iType, bool isTracked):
    parameterHoldingLabels_(label, std::vector<std::string>(), isTracked),
    type_(iType),
    isTracked_(isTracked) {
  }

  AllowedLabelsDescriptionBase::
  AllowedLabelsDescriptionBase(char const* label, ParameterTypes iType, bool isTracked):
    parameterHoldingLabels_(label, std::vector<std::string>(), isTracked),
    type_(iType),
    isTracked_(isTracked) {
  }


  void
  AllowedLabelsDescriptionBase::
  checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                             std::set<ParameterTypes> & parameterTypes,
                             std::set<ParameterTypes> & wildcardTypes) const {

    parameterHoldingLabels_.checkAndGetLabelsAndTypes(usedLabels, parameterTypes, wildcardTypes);
  }

  void
  AllowedLabelsDescriptionBase::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    parameterHoldingLabels_.validate(pset, validatedLabels, optional);
    if (parameterHoldingLabels_.exists(pset)) {
      std::vector<std::string> allowedLabels;
      if (isTracked()) {
        allowedLabels = pset.getParameter<std::vector<std::string> >(parameterHoldingLabels_.label());
      }
      else {
        allowedLabels = pset.getUntrackedParameter<std::vector<std::string> >(parameterHoldingLabels_.label());
      }
      for_all(allowedLabels, boost::bind(&AllowedLabelsDescriptionBase::validateAllowedLabel_,
                                         boost::cref(this),
                                         _1,
                                         boost::ref(pset),
                                         boost::ref(validatedLabels)));
    }
  }

  void
  AllowedLabelsDescriptionBase::
  writeCfi_(std::ostream & os,
            bool & startWithComma,
            int indentation,
            bool & wroteSomething) const {
    parameterHoldingLabels_.writeCfi(os, startWithComma, indentation, wroteSomething);
  }

  void
  AllowedLabelsDescriptionBase::
  print_(std::ostream & os,
         bool optional,
	 bool writeToCfi,
         DocFormatHelper & dfh)
  {
    if (dfh.pass() == 1) {

      dfh.indent(os);
      os << parameterHoldingLabels_.label() << " (list of allowed labels)";

      if (dfh.brief()) {

        if (optional)  os << " optional";

        if (!writeToCfi) os << " (do not write to cfi)";

        os << " see Section " << dfh.section() << "." << dfh.counter() << "\n";
      }
      // not brief
      else {

        os << "\n";
        dfh.indent2(os);

        if (optional)  os << "optional";
        if (!writeToCfi) os << " (do not write to cfi)";
        if (optional || !writeToCfi) {
          os << "\n";
          dfh.indent2(os);
        }

        os << "see Section " << dfh.section() << "." << dfh.counter() << "\n";

        if (!comment().empty()) {
          DocFormatHelper::wrapAndPrintText(os,
                                            comment(),
                                            dfh.startColumn2(),
                                            dfh.commentWidth());
        }
        os << "\n";
      }
    }
  }

  bool
  AllowedLabelsDescriptionBase::
  hasNestedContent_() {
    return true;
  }


  void
  AllowedLabelsDescriptionBase::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {
    printNestedContentBase_(os, optional, dfh);
    if (!dfh.brief()) os << "\n";
  }

  void
  AllowedLabelsDescriptionBase::
  printNestedContentBase_(std::ostream & os,
                          bool optional,
                          DocFormatHelper & dfh) {

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    printSpaces(os, indentation);
    os << "Section " << dfh.section() << "." << dfh.counter()
       << " " << parameterHoldingLabels_.label()
       << " - allowed labels description\n";
    printSpaces(os, indentation);
    os << "The following parameter contains a list of parameter labels\n";
    printSpaces(os, indentation);
    os << "which are allowed to be in the PSet\n";
    if (!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.init();
    new_dfh.setPass(1);
    parameterHoldingLabels_.print(os, optional, true, new_dfh);
    dfh.indent(os);
    os << "type of allowed parameters:";
    if (dfh.brief()) os << " ";
    else {
      os << "\n";
      dfh.indent2(os);
    } 
    if (!isTracked()) os << "untracked ";
    os << parameterTypeEnumToString(type()) << "\n";
  }

  bool
  AllowedLabelsDescriptionBase::
  exists_(ParameterSet const& pset) const {
    return parameterHoldingLabels_.exists(pset);
  }

  bool
  AllowedLabelsDescriptionBase::
  partiallyExists_(ParameterSet const& pset) const {
    return exists(pset);
  }

  int
  AllowedLabelsDescriptionBase::
  howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0;
  }
}
