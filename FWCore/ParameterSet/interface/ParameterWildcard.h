#ifndef FWCore_ParameterSet_ParameterWildcard_h
#define FWCore_ParameterSet_ParameterWildcard_h

#include "FWCore/ParameterSet/interface/ParameterWildcardBase.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <set>
#include <iosfwd>
#include <vector>

namespace edm {

  class ParameterSet;
  class VParameterSetEntry;
  class ParameterSetDescription;
  class DocFormatHelper;

  template<class T>
  class ParameterWildcard : public ParameterWildcardBase {

  public:

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked) :
      ParameterWildcardBase(ParameterTypeToEnum::toEnum<T>(), isTracked, criteria) {
      throwIfInvalidPattern(pattern);
    }

    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked) :
      ParameterWildcardBase(ParameterTypeToEnum::toEnum<T>(), isTracked, criteria) {
      throwIfInvalidPattern(pattern);
    }

    ~ParameterWildcard() override { }

    ParameterDescriptionNode* clone() const override {
      return new ParameterWildcard(*this);
    }

  private:
   
    void validate_(ParameterSet & pset,
                   std::set<std::string> & validatedLabels,
                   bool optional) const override {

      std::vector<std::string> parameterNames  = pset.getParameterNamesForType<T>(isTracked());
      validateMatchingNames(parameterNames, validatedLabels, optional);

    }

    bool exists_(ParameterSet const& pset) const override {

      if (criteria() == RequireZeroOrMore) return true;

      std::vector<std::string> parameterNames  = pset.getParameterNamesForType<T>(isTracked());

      if (criteria() == RequireAtLeastOne) return !parameterNames.empty();
      return parameterNames.size() == 1U;
    }

    // In the future may need to add a data member of type T to hold a default value
  };

  template<>
  class ParameterWildcard<ParameterSetDescription> : public ParameterWildcardBase {

  public:

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked);
    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked);

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked,
                      ParameterSetDescription const& desc);
    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked,
                      ParameterSetDescription const& desc);

    ~ParameterWildcard() override;

    ParameterDescriptionNode* clone() const override;

  private:
   
    void validate_(ParameterSet & pset,
                   std::set<std::string> & validatedLabels,
                   bool optional) const override;

    bool hasNestedContent_() const override;

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & helper) const override;

    bool exists_(ParameterSet const& pset) const override;

    void validateDescription(std::string const& parameterName, ParameterSet & pset) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
  class ParameterWildcard<std::vector<ParameterSet> > : public ParameterWildcardBase {

  public:

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked);
    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked);

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked,
                      ParameterSetDescription const& desc);
    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked,
                      ParameterSetDescription const& desc);

    ~ParameterWildcard() override;

    ParameterDescriptionNode* clone() const override;

  private:
   
    void validate_(ParameterSet & pset,
                   std::set<std::string> & validatedLabels,
                   bool optional) const override;

    bool hasNestedContent_() const override;

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    void validatePSetVector(std::string const& parameterName, ParameterSet & pset) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };
}
#endif
