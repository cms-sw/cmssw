
#include "FWCore/ParameterSet/interface/AllowedLabelsDescriptionBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "boost/bind.hpp"

namespace edm {

  AllowedLabelsDescriptionBase::~AllowedLabelsDescriptionBase() { }

  AllowedLabelsDescriptionBase::
  AllowedLabelsDescriptionBase(std::string const& label, bool isTracked):
    parameterHoldingLabels_(label, std::vector<std::string>(), isTracked),
    isTracked_(isTracked) {
  }

  AllowedLabelsDescriptionBase::
  AllowedLabelsDescriptionBase(char const* label, bool isTracked):
    parameterHoldingLabels_(label, std::vector<std::string>(), isTracked),
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
  howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0;
  }
}
