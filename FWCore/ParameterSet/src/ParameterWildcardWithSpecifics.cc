#include "FWCore/ParameterSet/interface/ParameterWildcardWithSpecifics.h"

#include "FWCore/ParameterSet/interface/DocFormatHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <cassert>
#include <iomanip>
#include <ostream>

namespace edm {

  ParameterWildcardWithSpecifics::ParameterWildcardWithSpecifics(
      std::string_view pattern,
      WildcardValidationCriteria criteria,
      bool isTracked,
      ParameterSetDescription const& desc,
      std::map<std::string, ParameterSetDescription> exceptions)
      : ParameterWildcardBase(k_PSet, isTracked, criteria), wildcardDesc_(desc), exceptions_(std::move(exceptions)) {
    throwIfInvalidPattern(std::string(pattern));
  }

  ParameterDescriptionNode* ParameterWildcardWithSpecifics::clone() const {
    return new ParameterWildcardWithSpecifics(*this);
  }

  void ParameterWildcardWithSpecifics::validate_(ParameterSet& pset,
                                                 std::set<std::string>& validatedLabels,
                                                 bool optional) const {
    std::vector<std::string> parameterNames = pset.getParameterNamesForType<ParameterSet>(isTracked());
    validateMatchingNames(parameterNames, validatedLabels, optional);

    for (auto const& name : parameterNames) {
      validateDescription(name, pset);
    }
    //inject exceptions if not already in the pset
    for (auto const& v : exceptions_) {
      if (std::find(parameterNames.begin(), parameterNames.end(), v.first) == parameterNames.end()) {
        if (isTracked()) {
          pset.addParameter<edm::ParameterSet>(v.first, edm::ParameterSet());
        } else {
          pset.addUntrackedParameter<edm::ParameterSet>(v.first, edm::ParameterSet());
        }
        validatedLabels.insert(v.first);
        validateDescription(v.first, pset);
      }
    }
  }

  void ParameterWildcardWithSpecifics::validateDescription(std::string const& parameterName, ParameterSet& pset) const {
    ParameterSet* containedPSet = pset.getPSetForUpdate(parameterName);
    auto itFound = exceptions_.find(parameterName);
    if (itFound != exceptions_.end()) {
      itFound->second.validate(*containedPSet);
    } else {
      wildcardDesc_.validate(*containedPSet);
    }
  }

  bool ParameterWildcardWithSpecifics::hasNestedContent_() const { return true; }

  void ParameterWildcardWithSpecifics::printNestedContent_(std::ostream& os,
                                                           bool /*optional*/,
                                                           DocFormatHelper& dfh) const {
    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    printSpaces(os, indentation);
    os << "Section " << dfh.section() << "." << dfh.counter() << " description of PSet matching wildcard:";
    os << "\n";
    if (!dfh.brief())
      os << "\n";

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    DocFormatHelper new_dfh(dfh);
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);

    wildcardDesc_.print(os, new_dfh);
    //NOTE: need to extend to also include the specific cases.
  }

  bool ParameterWildcardWithSpecifics::exists_(ParameterSet const& pset) const {
    if (criteria() == RequireZeroOrMore)
      return true;

    std::vector<std::string> parameterNames = pset.getParameterNamesForType<ParameterSet>(isTracked());

    if (criteria() == RequireAtLeastOne)
      return !parameterNames.empty();
    return parameterNames.size() == 1U;
  }

}  // namespace edm
