#ifndef FWCore_ParameterSet_XORGroupDescription_h
#define FWCore_ParameterSet_XORGroupDescription_h

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include "FWCore/Utilities/interface/value_ptr.h"

#include <memory>
#include <iosfwd>
#include <set>
#include <string>

namespace edm {

  class ParameterSet;
  class DocFormatHelper;

  class XORGroupDescription : public ParameterDescriptionNode {
  public:
    XORGroupDescription(ParameterDescriptionNode const& node_left,
                        ParameterDescriptionNode const& node_right);

    XORGroupDescription(std::unique_ptr<ParameterDescriptionNode> node_left,
                        ParameterDescriptionNode const& node_right);

    XORGroupDescription(ParameterDescriptionNode const& node_left,
                        std::unique_ptr<ParameterDescriptionNode> node_right);

    XORGroupDescription(std::unique_ptr<ParameterDescriptionNode> node_left,
                        std::unique_ptr<ParameterDescriptionNode> node_right);

    ParameterDescriptionNode* clone() const override {
      return new XORGroupDescription(*this);
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

    bool hasNestedContent_() const override {
      return true;
    }

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    bool partiallyExists_(ParameterSet const& pset) const override;

    int howManyXORSubNodesExist_(ParameterSet const& pset) const override;

    void throwMoreThanOneParameter() const;
    void throwAfterValidation() const;

    edm::value_ptr<ParameterDescriptionNode> node_left_;
    edm::value_ptr<ParameterDescriptionNode> node_right_;
  };
}
#endif
