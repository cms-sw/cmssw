#ifndef FWCore_ParameterSet_EmptyGroupDescription_h
#define FWCore_ParameterSet_EmptyGroupDescription_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include <iosfwd>
#include <set>
#include <string>

namespace edm {

  class ParameterSet;
  class DocFormatHelper;

  class EmptyGroupDescription : public ParameterDescriptionNode {
  public:
    EmptyGroupDescription();

    virtual ParameterDescriptionNode* clone() const {
      return new EmptyGroupDescription(*this);
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

    virtual void print_(std::ostream & os,
                        bool optional,
                        bool writeToCfi,
                        DocFormatHelper & dfh);

    virtual bool exists_(ParameterSet const& pset) const;

    virtual bool partiallyExists_(ParameterSet const& pset) const;

    virtual int howManyXORSubNodesExist_(ParameterSet const& pset) const;
  };
}
#endif
