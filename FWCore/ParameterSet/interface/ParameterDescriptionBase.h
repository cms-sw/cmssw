
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
#include <vector>
#include <set>
#include <iosfwd>

namespace edm {

  class ParameterSetDescription;

  class ParameterDescriptionBase : public ParameterDescriptionNode 
  {
  public:
    virtual ~ParameterDescriptionBase();

    std::string const& label() const { return label_; }
    ParameterTypes type() const { return type_; }
    bool isTracked() const { return isTracked_; }

    virtual ParameterSetDescription const* parameterSetDescription() const { return 0; }
    virtual ParameterSetDescription * parameterSetDescription() { return 0; }

    virtual std::vector<ParameterSetDescription> const* parameterSetDescriptions() const { return 0; }
    virtual std::vector<ParameterSetDescription> * parameterSetDescriptions() { return 0; }

    void throwParameterWrongTrackiness() const;
    void throwParameterWrongType() const;

  protected:
    ParameterDescriptionBase(std::string const& iLabel,
                             ParameterTypes iType,
                             bool isTracked
                            );

    ParameterDescriptionBase(char const* iLabel,
                             ParameterTypes iType,
                             bool isTracked
                            );

  private:

    virtual void checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                                            std::set<ParameterTypes> & parameterTypes,
                                            std::set<ParameterTypes> & wildcardTypes) const;

    virtual void writeCfi_(std::ostream & os,
                           bool & startWithComma,
                           int indentation,
                           bool & wroteSomething) const;

    virtual bool partiallyExists_(ParameterSet const& pset) const;

    virtual int howManyExclusiveOrSubNodesExist_(ParameterSet const& pset) const;

    virtual void writeCfi_(std::ostream & os, int indentation) const = 0;

    std::string label_;
    ParameterTypes type_;
    bool isTracked_;
  };
}
#endif
