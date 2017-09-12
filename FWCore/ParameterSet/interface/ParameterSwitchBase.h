#ifndef FWCore_ParameterSet_ParameterSwitchBase_h
#define FWCore_ParameterSet_ParameterSwitchBase_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <set>
#include <utility>
#include <iosfwd>

namespace edm {

  class DocFormatHelper;

  class ParameterSwitchBase : public ParameterDescriptionNode {
  public:
    ~ParameterSwitchBase() override;

  protected:
    void insertAndCheckLabels(std::string const& switchLabel,
                              std::set<std::string> & usedLabels,
                              std::set<std::string> & labels) const;

    void insertAndCheckTypes(ParameterTypes switchType,
                             std::set<ParameterTypes> const& caseParameterTypes,
                             std::set<ParameterTypes> const& caseWildcardTypes,
                             std::set<ParameterTypes> & parameterTypes,
                             std::set<ParameterTypes> & wildcardTypes) const;

    void throwDuplicateCaseValues(std::string const& switchLabel) const;

    void throwNoCaseForDefault(std::string const& switchLabel) const;

    void throwNoCaseForSwitchValue(std::string const& message) const;

    void printBase(std::ostream & os,
                   bool optional,
                   bool writeToCfi,
                   DocFormatHelper & dfh,
                   std::string const& switchLabel,
                   bool isTracked,
                   std::string const& typeString) const;

    bool hasNestedContent_() const override;

    void printNestedContentBase(std::ostream & os,
                                DocFormatHelper & dfh,
                                DocFormatHelper & new_dfh,
                                std::string const& switchLabel) const;

    template <typename T>
    static void printCaseT(std::pair<T, edm::value_ptr<ParameterDescriptionNode> > const& p,
                           std::ostream & os,
                           bool optional,
                           DocFormatHelper & dfh,
                           std::string const& switchLabel) {
      ParameterSwitchBase::printCase(p, os, optional, dfh, switchLabel);
    }

  private:

    static void printCase(std::pair<bool, edm::value_ptr<ParameterDescriptionNode> > const& p,
                          std::ostream & os,
                          bool optional,
                          DocFormatHelper & dfh,
                          std::string const& switchLabel);

    static void printCase(std::pair<int, edm::value_ptr<ParameterDescriptionNode> > const& p,
                          std::ostream & os,
                          bool optional,
                          DocFormatHelper & dfh,
                          std::string const& switchLabel);

    static void printCase(std::pair<std::string, edm::value_ptr<ParameterDescriptionNode> > const& p,
                          std::ostream & os,
                          bool optional,
                          DocFormatHelper & dfh,
                          std::string const& switchLabel);

    bool partiallyExists_(ParameterSet const& pset) const override;

    int howManyXORSubNodesExist_(ParameterSet const& pset) const override;

  };
}

#endif
