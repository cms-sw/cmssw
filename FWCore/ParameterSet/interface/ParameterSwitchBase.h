#ifndef FWCore_ParameterSet_ParameterSwitchBase_h
#define FWCore_ParameterSet_ParameterSwitchBase_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include <string>
#include <set>

namespace edm {

  class ParameterSwitchBase : public ParameterDescriptionNode {
  public:
    virtual ~ParameterSwitchBase();

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

  private:

    virtual bool partiallyExists_(ParameterSet const& pset) const;

    virtual int howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const;

  };
}

#endif
