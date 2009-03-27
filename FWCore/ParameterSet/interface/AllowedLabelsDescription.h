#ifndef FWCore_ParameterSet_AllowedLabelsDescription_h
#define FWCore_ParameterSet_AllowedLabelsDescription_h

#include "FWCore/ParameterSet/interface/AllowedLabelsDescriptionBase.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <set>
#include <vector>

namespace edm {

  class VParameterSetEntry;
  class ParameterSetDescription;

  template<class T>
  class AllowedLabelsDescription : public AllowedLabelsDescriptionBase {

  public:
    AllowedLabelsDescription(std::string const& label,
                             bool isTracked) :
      AllowedLabelsDescriptionBase(label, isTracked)
    {              
    }

    AllowedLabelsDescription(char const* label,
                             bool isTracked) :
      AllowedLabelsDescriptionBase(label, isTracked)
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

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
    class AllowedLabelsDescription<std::vector<ParameterSetDescription> > : public AllowedLabelsDescriptionBase {

  public:
    AllowedLabelsDescription(std::string const& label,
                             bool isTracked);

    AllowedLabelsDescription(char const* label,
                             bool isTracked);

    AllowedLabelsDescription(std::string const& label,
                             std::vector<ParameterSetDescription> const& value,
                             bool isTracked);

    AllowedLabelsDescription(char const* label,
                             std::vector<ParameterSetDescription> const& value,
                             bool isTracked);

    virtual ParameterDescriptionNode* clone() const;

  private:

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const;

    void
    validateDescription(ParameterSetDescription const& psetDescription,
                        VParameterSetEntry * vpsetEntry,
                        int & i) const;

    value_ptr<std::vector<ParameterSetDescription> > vPsetDesc_;
  };
}
#endif
