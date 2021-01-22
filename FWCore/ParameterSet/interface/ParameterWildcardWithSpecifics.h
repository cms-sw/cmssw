
#ifndef FWCore_ParameterSet_ParameterWildcardWithSpecifics_h
#define FWCore_ParameterSet_ParameterWildcardWithSpecifics_h

#include <string>
#include <map>
#include "FWCore/ParameterSet/interface/ParameterWildcardBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  class ParameterSet;
  class DocFormatHelper;

  class ParameterWildcardWithSpecifics : public ParameterWildcardBase {
  public:
    ParameterWildcardWithSpecifics(std::string_view,
                                   WildcardValidationCriteria criteria,
                                   bool isTracked,
                                   ParameterSetDescription const& desc,
                                   std::map<std::string, ParameterSetDescription> exceptions);

    ParameterDescriptionNode* clone() const override;

  private:
    void validate_(ParameterSet& pset, std::set<std::string>& validatedLabels, bool optional) const override;

    bool hasNestedContent_() const override;

    void printNestedContent_(std::ostream& os, bool optional, DocFormatHelper& dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    void validatePSetVector(std::string const& parameterName, ParameterSet& pset) const;

    void validateDescription(std::string const& parameterName, ParameterSet& pset) const;

    ParameterSetDescription wildcardDesc_;
    std::map<std::string, ParameterSetDescription> exceptions_;
  };
}  // namespace edm
#endif
