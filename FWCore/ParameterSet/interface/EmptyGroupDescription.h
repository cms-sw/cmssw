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

    ParameterDescriptionNode* clone() const override {
      return new EmptyGroupDescription(*this);
    }

  private:

    void checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                                    std::set<ParameterTypes> & parameterTypes,
                                    std::set<ParameterTypes> & wildcardTypes) const override;

    void validate_(ParameterSet & pset,
                   std::set<std::string> & validatedLabels,
                   bool optional) const override;

    void writeCfi_(std::ostream & os,
                   bool & startWithComma,
                   int indentation,
                   bool & wroteSomething) const override;

    void print_(std::ostream & os,
                bool optional,
                bool writeToCfi,
                DocFormatHelper & dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    bool partiallyExists_(ParameterSet const& pset) const override;

    int howManyXORSubNodesExist_(ParameterSet const& pset) const override;
  };
}
#endif
