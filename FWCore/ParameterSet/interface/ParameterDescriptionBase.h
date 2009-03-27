
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

 In addition to whatever you need to do to add a new type to the
 ParameterSet code, you need to do the following to add
 a new type to the ParameterSetDescription code:
 1.  add a value to the enumeration ParameterTypes (ParameterDescriptionBase.h)
 2.  add new TYPE_TO_NAME and TYPE_TO_ENUM macros (ParameterDescriptionBase.cc)
 3.  add declaration of writeValueToCfi function to ParameterDescription.h
 (Two of them if a vector of the type is also allowed)
 4.  define writeValueToCfi in ParameterDescription.cc
 5.  Consider whether you need a specialization of writeSingleValue and
 writeValueInVector in ParameterDescription.cc. The first is needed
 if operator<< for the new type does not print the correct format for a cfi.
 The second is needed if the format in a vector is different from the format
 when a single value is not in a vector.
 6.  add parameters of that type and vectors of that type to
 FWCore/Integration/test/ProducerWithPSetDesc.cc
 7.  Check and update the reference file in
 FWCore/Integration/test/unit_test_outputs/testProducerWithPsetDesc_cfi.py
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
