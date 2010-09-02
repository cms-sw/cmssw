#include "FWCore/ParameterSet/interface/ParameterWildcard.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include "boost/bind.hpp"

#include <cassert>
#include <ostream>
#include <iomanip>

namespace edm {

  ParameterWildcard<ParameterSetDescription>::
  ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked) :
    ParameterWildcardBase(k_PSet, isTracked, criteria),
    psetDesc_()
  {
    throwIfInvalidPattern(pattern);
  }

  ParameterWildcard<ParameterSetDescription>::
  ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked) :
    ParameterWildcardBase(k_PSet, isTracked, criteria),
    psetDesc_()
  {
    throwIfInvalidPattern(pattern);
  }

  ParameterWildcard<ParameterSetDescription>::
  ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked, ParameterSetDescription const& desc) :
    ParameterWildcardBase(k_PSet, isTracked, criteria),
    psetDesc_(new ParameterSetDescription(desc))
  {
    throwIfInvalidPattern(pattern);
  }


  ParameterWildcard<ParameterSetDescription>::
  ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked, ParameterSetDescription const& desc) :
    ParameterWildcardBase(k_PSet, isTracked, criteria),
    psetDesc_(new ParameterSetDescription(desc))
  {
    throwIfInvalidPattern(pattern);
  }


  ParameterWildcard<ParameterSetDescription>::
  ~ParameterWildcard() { }

  ParameterDescriptionNode*
  ParameterWildcard<ParameterSetDescription>::
  clone() const {
    return new ParameterWildcard(*this);
  }

  void
  ParameterWildcard<ParameterSetDescription>::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    std::vector<std::string> parameterNames  = pset.getParameterNamesForType<ParameterSet>(isTracked());
    validateMatchingNames(parameterNames, validatedLabels, optional);

    if (psetDesc_) {
      for_all(parameterNames,
              boost::bind(&ParameterWildcard<ParameterSetDescription>::validateDescription,
                          boost::cref(this),
                          _1,
                          boost::ref(pset)));
    }
  }

  void
  ParameterWildcard<ParameterSetDescription>::
  validateDescription(std::string const& parameterName,
                      ParameterSet & pset) const {
    ParameterSet * containedPSet = pset.getPSetForUpdate(parameterName);
    psetDesc_->validate(*containedPSet);
  }

  bool
  ParameterWildcard<ParameterSetDescription>::
  hasNestedContent_() {
    if (psetDesc_) return true;
    return false;
  }

  void
  ParameterWildcard<ParameterSetDescription>::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    printSpaces(os, indentation);
    os << "Section " << dfh.section() << "." << dfh.counter()
       << " description of PSet matching wildcard:";
    os << "\n";
    if (!dfh.brief()) os << "\n";

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    DocFormatHelper new_dfh(dfh);
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);

    psetDesc_->print(os, new_dfh);
  }

  bool
  ParameterWildcard<ParameterSetDescription>::
  exists_(ParameterSet const& pset) const {

    if (criteria() == RequireZeroOrMore) return true;

    std::vector<std::string> parameterNames  = pset.getParameterNamesForType<ParameterSet>(isTracked());

    if (criteria() == RequireAtLeastOne) return parameterNames.size() >= 1U;
    return parameterNames.size() == 1U;
  }

// -------------------------------------------------------------------------

  ParameterWildcard<std::vector<ParameterSet> >::
  ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    psetDesc_()
  {
    throwIfInvalidPattern(pattern);
  }

  ParameterWildcard<std::vector<ParameterSet> >::
  ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    psetDesc_()
  {
    throwIfInvalidPattern(pattern);
  }

  ParameterWildcard<std::vector<ParameterSet> >::
  ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked,
                    ParameterSetDescription const& desc) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    psetDesc_(new ParameterSetDescription(desc))
  {
    throwIfInvalidPattern(pattern);
  }


  ParameterWildcard<std::vector<ParameterSet> >::
  ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked,
                    ParameterSetDescription const& desc) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    psetDesc_(new ParameterSetDescription(desc))
  {
    throwIfInvalidPattern(pattern);
  }


  ParameterWildcard<std::vector<ParameterSet> >::
  ~ParameterWildcard() { }

  ParameterDescriptionNode*
  ParameterWildcard<std::vector<ParameterSet> >::
  clone() const {
    return new ParameterWildcard(*this);
  }

  void
  ParameterWildcard<std::vector<ParameterSet> >::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    std::vector<std::string> parameterNames  = pset.getParameterNamesForType<std::vector<ParameterSet> >(isTracked());
    validateMatchingNames(parameterNames, validatedLabels, optional);

    if (psetDesc_) {
      for_all(parameterNames,
              boost::bind(&ParameterWildcard<std::vector<ParameterSet> >::validatePSetVector,
                          boost::cref(this),
                          _1,
                          boost::ref(pset)));
    }
  }

  void
  ParameterWildcard<std::vector<ParameterSet> >::
  validatePSetVector(std::string const& parameterName, ParameterSet & pset) const {
    VParameterSetEntry * vpsetEntry = pset.getPSetVectorForUpdate(parameterName);
    assert(vpsetEntry);
    for (unsigned i = 0; i < vpsetEntry->size(); ++i) {
      psetDesc_->validate(vpsetEntry->psetInVector(i));
    }
  }

  bool
  ParameterWildcard<std::vector<ParameterSet> >::
  hasNestedContent_() {
    if (psetDesc_) return true;
    return false;
  }

  void
  ParameterWildcard<std::vector<ParameterSet> >::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    printSpaces(os, indentation);
    os << "Section " << dfh.section() << "." << dfh.counter()
       << " description used to validate all PSets which are in the VPSet matching the wildcard:";
    os << "\n";
    if (!dfh.brief()) os << "\n";

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    DocFormatHelper new_dfh(dfh);
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);

    psetDesc_->print(os, new_dfh);
  }

  bool
  ParameterWildcard<std::vector<ParameterSet> >::
  exists_(ParameterSet const& pset) const {

    if (criteria() == RequireZeroOrMore) return true;

    std::vector<std::string> parameterNames  = pset.getParameterNamesForType<std::vector<ParameterSet> >(isTracked());

    if (criteria() == RequireAtLeastOne) return parameterNames.size() >= 1U;
    return parameterNames.size() == 1U;
  }
}
