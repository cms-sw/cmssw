
#ifndef FWCore_ParameterSet_AllowedLabelsDescription_h
#define FWCore_ParameterSet_AllowedLabelsDescription_h

#include "FWCore/ParameterSet/interface/AllowedLabelsDescriptionBase.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <set>
#include <vector>
#include <iosfwd>

namespace edm {

  class VParameterSetEntry;
  class ParameterSetDescription;
  class DocFormatHelper;

  template<class T>
  class AllowedLabelsDescription : public AllowedLabelsDescriptionBase {

  public:
    AllowedLabelsDescription(std::string const& label,
                             bool isTracked) :
      AllowedLabelsDescriptionBase(label, ParameterTypeToEnum::toEnum<T>(), isTracked)
    {              
    }

    AllowedLabelsDescription(char const* label,
                             bool isTracked) :
      AllowedLabelsDescriptionBase(label, ParameterTypeToEnum::toEnum<T>(), isTracked)
    {
    }

    virtual ParameterDescriptionNode* clone() const {
      return new AllowedLabelsDescription(*this);
    }

  private:

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const {
      if (pset.existsAs<T>(allowedLabel, isTracked())) {
        validatedLabels.insert(allowedLabel);
      }
    }
  };

  template<>
  class AllowedLabelsDescription<ParameterSetDescription> : public AllowedLabelsDescriptionBase {

  public:
    AllowedLabelsDescription(std::string const& label,
                             bool isTracked);

    AllowedLabelsDescription(char const* label,
                             bool isTracked);

    AllowedLabelsDescription(std::string const& label,
                             ParameterSetDescription const& value,
                             bool isTracked);

    AllowedLabelsDescription(char const* label,
                             ParameterSetDescription const& value,
                             bool isTracked);

    virtual ParameterDescriptionNode* clone() const;

  private:

    virtual void printNestedContent_(std::ostream & os,
                                     bool optional,
                                     DocFormatHelper & helper) const;

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
  class AllowedLabelsDescription<std::vector<ParameterSet> > : public AllowedLabelsDescriptionBase {

  public:
    AllowedLabelsDescription(std::string const& label,
                             bool isTracked);

    AllowedLabelsDescription(char const* label,
                             bool isTracked);

    AllowedLabelsDescription(std::string const& label,
                             ParameterSetDescription const& value,
                             bool isTracked);

    AllowedLabelsDescription(char const* label,
                             ParameterSetDescription const& value,
                             bool isTracked);

    virtual ParameterDescriptionNode* clone() const;

  private:

    virtual void printNestedContent_(std::ostream & os,
                                     bool optional,
                                     DocFormatHelper & helper) const;

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };
}
#endif
