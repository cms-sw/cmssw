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

    ~AllowedLabelsDescriptionBase() override;

    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }

  protected:

    AllowedLabelsDescriptionBase(std::string const& label, ParameterTypes iType, bool isTracked);

    AllowedLabelsDescriptionBase(char const* label, ParameterTypes iType, bool isTracked);

    void printNestedContentBase_(std::ostream & os,
                                 bool optional,
                                 DocFormatHelper & dfh) const;

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

    bool hasNestedContent_() const override;

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    bool partiallyExists_(ParameterSet const& pset) const override;

    int howManyXORSubNodesExist_(ParameterSet const& pset) const override;

    virtual void validateAllowedLabel_(std::string const& allowedLabel,
                                       ParameterSet & pset,
                                       std::set<std::string> & validatedLabels) const = 0;

    ParameterDescription<std::vector<std::string> > parameterHoldingLabels_;
    ParameterTypes type_;
    bool isTracked_;
  };
}

#endif
