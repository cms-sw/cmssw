
#ifndef FWCore_ParameterSet_ParameterDescriptionBase_h
#define FWCore_ParameterSet_ParameterDescriptionBase_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescriptionBase
//
/**\class ParameterDescriptionBase ParameterDescriptionBase.h FWCore/ParameterSet/interface/ParameterDescriptionBase.h

 Description: Base class for a description of one parameter in a ParameterSet

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:33:46 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include <string>
#include <set>
#include <iosfwd>

namespace edm {

  class ParameterSetDescription;
  class DocFormatHelper;

  class ParameterDescriptionBase : public ParameterDescriptionNode 
  {
  public:
    ~ParameterDescriptionBase() override;

    std::string const& label() const { return label_; }
    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }
    bool hasDefault() const { return hasDefault_; }

    virtual ParameterSetDescription const* parameterSetDescription() const { return nullptr; }
    virtual ParameterSetDescription * parameterSetDescription() { return nullptr; }

  protected:
    void throwParameterWrongTrackiness() const;
    void throwParameterWrongType() const;
    void throwMissingRequiredNoDefault() const;

    ParameterDescriptionBase(std::string const& iLabel,
                             ParameterTypes iType,
                             bool isTracked,
                             bool hasDefault,
                             Comment const& iComment
                            );

    ParameterDescriptionBase(char const* iLabel,
                             ParameterTypes iType,
                             bool isTracked,
                             bool hasDefault,
                             Comment const& iComment
                            );

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

    bool partiallyExists_(ParameterSet const& pset) const override;

    int howManyXORSubNodesExist_(ParameterSet const& pset) const override;

    virtual void writeCfi_(std::ostream & os, int indentation) const = 0;

    virtual void writeDoc_(std::ostream & os, int indentation) const = 0;

    void print_(std::ostream & os,
                bool optional,
                bool writeToCfi,
                DocFormatHelper & dfh) const override;

    virtual void printDefault_(std::ostream & os,
                               bool writeToCfi,
                               DocFormatHelper & dfh) const;

    void printNestedContent_(std::ostream & os,
                             bool optional,
                             DocFormatHelper & dfh) const override;

    using ParameterDescriptionNode::exists_;
    virtual bool exists_(ParameterSet const& pset, bool isTracked) const = 0;

    virtual void insertDefault_(ParameterSet & pset) const = 0;

    std::string label_;
    ParameterTypes type_;
    bool isTracked_;
    bool hasDefault_;
  };
}
#endif
