#ifndef FWCore_ParameterSet_AllowedLabelsDescriptionBase_h
#define FWCore_ParameterSet_AllowedLabelsDescriptionBase_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"

#include <vector>
#include <string>
#include <set>
#include <iosfwd>

namespace edm {

  class ParameterSet;

  class AllowedLabelsDescriptionBase : public ParameterDescriptionNode {
  public:

    virtual ~AllowedLabelsDescriptionBase();

    bool isTracked() const { return isTracked_; }

  protected:

    AllowedLabelsDescriptionBase(std::string const& label, bool isTracked);

    AllowedLabelsDescriptionBase(char const* label, bool isTracked);

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

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const = 0;

    ParameterDescription<std::vector<std::string> > parameterHoldingLabels_;
    bool isTracked_;
  };
}

#endif
