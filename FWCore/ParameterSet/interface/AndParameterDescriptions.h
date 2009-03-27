#ifndef FWCore_ParameterSet_AndParameterDescriptions_h
#define FWCore_ParameterSet_AndParameterDescriptions_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include "FWCore/Utilities/interface/value_ptr.h"

#include <memory>
#include <iosfwd>
#include <set>
#include <string>

namespace edm {

  class ParameterSet;

  class AndParameterDescriptions : public ParameterDescriptionNode {
  public:
    AndParameterDescriptions(ParameterDescriptionNode const& node_left,
                             ParameterDescriptionNode const& node_right);

    AndParameterDescriptions(std::auto_ptr<ParameterDescriptionNode> node_left,
                             ParameterDescriptionNode const& node_right);

    AndParameterDescriptions(ParameterDescriptionNode const& node_left,
                             std::auto_ptr<ParameterDescriptionNode> node_right);

    AndParameterDescriptions(std::auto_ptr<ParameterDescriptionNode> node_left,
                             std::auto_ptr<ParameterDescriptionNode> node_right);

    virtual ParameterDescriptionNode* clone() const {
      return new AndParameterDescriptions(*this);
    }

  private:

    virtual void checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                                            std::set<ParameterTypes> & parameterTypes,
                                            std::set<ParameterTypes> & wildcardTypes) const;

    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const;

    virtual void writeCfi_(std::ostream & os,
                           bool & startWithComma,
                           int indentation,
                           bool & wroteSomething) const;

    virtual bool exists_(ParameterSet const& pset) const;

    virtual bool partiallyExists_(ParameterSet const& pset) const;

    virtual int howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const;


    void throwIfDuplicateLabels(std::set<std::string> const& labelsLeft,
                                std::set<std::string> const& labelsRight) const;

    void throwIfDuplicateTypes(std::set<ParameterTypes> const& types1,
                               std::set<ParameterTypes> const& types2) const;

    edm::value_ptr<ParameterDescriptionNode> node_left_;
    edm::value_ptr<ParameterDescriptionNode> node_right_;
  };
}
#endif
