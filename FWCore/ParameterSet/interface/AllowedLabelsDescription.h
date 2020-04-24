
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

    ParameterDescriptionNode* clone() const override {
      return new AllowedLabelsDescription(*this);
    }

  private:

    void validateAllowedLabel_(std::string const& allowedLabel,
                               ParameterSet & pset,
                               std::set<std::string> & validatedLabels) const override {
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

    ParameterDescriptionNode* clone() const override;

  private:

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & helper) const override;

    void validateAllowedLabel_(std::string const& allowedLabel,
                               ParameterSet & pset,
                               std::set<std::string> & validatedLabels) const override;

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

    ParameterDescriptionNode* clone() const override;

  private:

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & helper) const override;

    void validateAllowedLabel_(std::string const& allowedLabel,
                               ParameterSet & pset,
                               std::set<std::string> & validatedLabels) const override;

    value_ptr<ParameterSetDescription> psetDesc_;
  };
}
#endif
