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

    virtual ~ParameterWildcard() { }

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterWildcard(*this);
    }

  private:
   
    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const {

      std::vector<std::string> parameterNames  = pset.getParameterNamesForType<T>(isTracked());
      validateMatchingNames(parameterNames, validatedLabels, optional);

    }

    virtual bool exists_(ParameterSet const& pset) const {

      if (criteria() == RequireZeroOrMore) return true;

      std::vector<std::string> parameterNames  = pset.getParameterNamesForType<T>(isTracked());

      if (criteria() == RequireAtLeastOne) return parameterNames.size() >= 1U;
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

    virtual ~ParameterWildcard();

    virtual ParameterDescriptionNode* clone() const;

  private:
   
    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const;

    virtual bool exists_(ParameterSet const& pset) const;

    void validateDescription(std::string const& parameterName, ParameterSet & pset) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
  class ParameterWildcard<std::vector<ParameterSetDescription> > : public ParameterWildcardBase {

  public:

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked);
    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked);

    ParameterWildcard(std::string const& pattern, WildcardValidationCriteria criteria, bool isTracked,
                      std::vector<ParameterSetDescription> const& desc);
    ParameterWildcard(char const* pattern, WildcardValidationCriteria criteria, bool isTracked,
                      std::vector<ParameterSetDescription> const& desc);

    virtual ~ParameterWildcard();

    virtual ParameterDescriptionNode* clone() const;

  private:
   
    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const;

    virtual bool exists_(ParameterSet const& pset) const;

    void validateDescriptionVector(std::string const& parameterName, ParameterSet & pset) const;

    void validateDescription(ParameterSetDescription const& psetDescription,
                             VParameterSetEntry * vpsetEntry,
                             int & i) const;

    value_ptr<std::vector<ParameterSetDescription> > vPsetDesc_;
  };
}
#endif
