
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
    virtual ~ParameterDescriptionBase();

    std::string const& label() const { return label_; }
    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }
    bool hasDefault() const { return hasDefault_; }

    virtual ParameterSetDescription const* parameterSetDescription() const { return 0; }
    virtual ParameterSetDescription * parameterSetDescription() { return 0; }

  protected:
    void throwParameterWrongTrackiness() const;
    void throwParameterWrongType() const;
    void throwMissingRequiredNoDefault() const;

    ParameterDescriptionBase(std::string const& iLabel,
                             ParameterTypes iType,
                             bool isTracked,
                             bool hasDefault
                            );

    ParameterDescriptionBase(char const* iLabel,
                             ParameterTypes iType,
                             bool isTracked,
                             bool hasDefault
                            );

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

    virtual bool partiallyExists_(ParameterSet const& pset) const;

    virtual int howManyXORSubNodesExist_(ParameterSet const& pset) const;

    virtual void writeCfi_(std::ostream & os, int indentation) const = 0;

    virtual void writeDoc_(std::ostream & os, int indentation) const = 0;

    virtual void print_(std::ostream & os,
                        bool optional,
                        bool writeToCfi,
                        DocFormatHelper & dfh);

    virtual void printDefault_(std::ostream & os,
                                 bool writeToCfi,
                                 DocFormatHelper & dfh);

    virtual void printNestedContent_(std::ostream & os,
                                     bool optional,
                                     DocFormatHelper & dfh);

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
