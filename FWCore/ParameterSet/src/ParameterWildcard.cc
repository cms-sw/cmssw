#include "FWCore/ParameterSet/interface/ParameterWildcard.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/Utilities/interface/EDMException.h"
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

    os << std::setfill(' ') << std::setw(indentation) << "";
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

  ParameterWildcard<std::vector<ParameterSetDescription> >::
  ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    vPsetDesc_()
  {
    throwIfInvalidPattern(pattern);
  }

  ParameterWildcard<std::vector<ParameterSetDescription> >::
  ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    vPsetDesc_()
  {
    throwIfInvalidPattern(pattern);
  }

  ParameterWildcard<std::vector<ParameterSetDescription> >::
  ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked,
                    std::vector<ParameterSetDescription> const& desc) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    vPsetDesc_(new std::vector<ParameterSetDescription>(desc))
  {
    throwIfInvalidPattern(pattern);
  }


  ParameterWildcard<std::vector<ParameterSetDescription> >::
  ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked,
                    std::vector<ParameterSetDescription> const& desc) :
    ParameterWildcardBase(k_VPSet, isTracked, criteria),
    vPsetDesc_(new std::vector<ParameterSetDescription>(desc))
  {
    throwIfInvalidPattern(pattern);
  }


  ParameterWildcard<std::vector<ParameterSetDescription> >::
  ~ParameterWildcard() { }

  ParameterDescriptionNode*
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  clone() const {
    return new ParameterWildcard(*this);
  }

  void
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    std::vector<std::string> parameterNames  = pset.getParameterNamesForType<std::vector<ParameterSet> >(isTracked());
    validateMatchingNames(parameterNames, validatedLabels, optional);

    if (vPsetDesc_) {
      for_all(parameterNames,
              boost::bind(&ParameterWildcard<std::vector<ParameterSetDescription> >::validateDescriptionVector,
                          boost::cref(this),
                          _1,
                          boost::ref(pset)));
    }
  }

  void
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  validateDescriptionVector(std::string const& parameterName, ParameterSet & pset) const {
    VParameterSetEntry * vpsetEntry = pset.getPSetVectorForUpdate(parameterName);
    assert(vpsetEntry);
    if (vpsetEntry->size() != vPsetDesc_->size()) {
      throw edm::Exception(errors::Configuration)
        << "Unexpected number of ParameterSets in vector of parameter sets named \"" << parameterName << "\".";
    }
    int i = 0;
    for_all(*vPsetDesc_,
            boost::bind(&ParameterWildcard<std::vector<ParameterSetDescription> >::validateDescription,
                        boost::cref(this),
                        _1,
                        vpsetEntry,
                        boost::ref(i)));
  }

  void
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  validateDescription(ParameterSetDescription const& psetDescription,
                      VParameterSetEntry * vpsetEntry,
                      int & i) const {
    psetDescription.validate(vpsetEntry->psetInVector(i));
    ++i;
  }

  bool
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  hasNestedContent_() {
    if (vPsetDesc_) return true;
    return false;
  }

  void
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    os << std::setfill(' ') << std::setw(indentation) << "";
    os << "Section " << dfh.section() << "." << dfh.counter()
       << " description of VPSet matching wildcard:\n";

    for (unsigned i = 1; i <= vPsetDesc_->size(); ++i) {
      os << std::setfill(' ') 
         << std::setw(indentation + DocFormatHelper::offsetSectionContent())
         << "";
      os << "[" << (i - 1) << "]: see Section " << dfh.section() << "." << dfh.counter()
         << "." << i << "\n";
    }
    if (!dfh.brief()) os << "\n";

    for (unsigned i = 1; i <= vPsetDesc_->size(); ++i) {

      std::stringstream ss;
      ss << dfh.section() << "." << dfh.counter() << "." << i;
      std::string newSection = ss.str();

      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "Section " << newSection << " description of PSet:\n";
      if (!dfh.brief()) os << "\n";

      DocFormatHelper new_dfh(dfh);
      new_dfh.setSection(newSection);
      new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
      new_dfh.setParent(DocFormatHelper::OTHER);
      (*vPsetDesc_)[i - 1].print(os, new_dfh);
    }
  }

  bool
  ParameterWildcard<std::vector<ParameterSetDescription> >::
  exists_(ParameterSet const& pset) const {

    if (criteria() == RequireZeroOrMore) return true;

    std::vector<std::string> parameterNames  = pset.getParameterNamesForType<std::vector<ParameterSet> >(isTracked());

    if (criteria() == RequireAtLeastOne) return parameterNames.size() >= 1U;
    return parameterNames.size() == 1U;
  }
}
