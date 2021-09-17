
#ifndef FWCore_ParameterSet_ParameterWildcardBase_h
#define FWCore_ParameterSet_ParameterWildcardBase_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include <set>
#include <string>
#include <iosfwd>
#include <vector>

namespace edm {

  class ParameterSet;
  class DocFormatHelper;

  enum WildcardValidationCriteria { RequireZeroOrMore, RequireAtLeastOne, RequireExactlyOne };

  class ParameterWildcardBase : public ParameterDescriptionNode {
  public:
    ~ParameterWildcardBase() override;

    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }
    WildcardValidationCriteria criteria() const { return criteria_; }

  protected:
    ParameterWildcardBase(ParameterTypes iType, bool isTracked, WildcardValidationCriteria criteria);

    void throwIfInvalidPattern(char const* pattern) const;
    void throwIfInvalidPattern(std::string const& pattern) const;

    void validateMatchingNames(std::vector<std::string> const& matchingNames,
                               std::set<std::string>& validatedLabels,
                               bool optional) const;

  private:
    void checkAndGetLabelsAndTypes_(std::set<std::string>& usedLabels,
                                    std::set<ParameterTypes>& parameterTypes,
                                    std::set<ParameterTypes>& wildcardTypes) const override;

    void writeCfi_(
        std::ostream& os, bool optional, bool& startWithComma, int indentation, bool& wroteSomething) const override;

    void print_(std::ostream& os, bool optional, bool writeToCfi, DocFormatHelper& dfh) const override;

    bool partiallyExists_(ParameterSet const& pset) const override;

    int howManyXORSubNodesExist_(ParameterSet const& pset) const override;

    ParameterTypes type_;
    bool isTracked_;
    WildcardValidationCriteria criteria_;

    // In the future we may want to add a string for the label if
    // default values are added to be inserted into a ParameterSet
    // when missing.

    // In the future we may want to add a string for the wildcard
    // pattern if we implement regular expressions, globbing, or some
    // other kind of wildcard patterns other than "*".
  };
}  // namespace edm
#endif
