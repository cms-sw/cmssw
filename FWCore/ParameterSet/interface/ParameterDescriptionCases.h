#ifndef FWCore_ParameterSet_ParameterDescriptionCases_h
#define FWCore_ParameterSet_ParameterDescriptionCases_h

// This class is used to store temporary objects created when
// building ParameterSwitch's in a ParameterSetDescription.
// It gets created while evaluating an expression something
// like
//
// parameterSetDescription.ifValue( ParameterDescription<int>("switch", 0),
//                                  0 >> ParameterDescription<int>("label1", 11) or
//                                  1 >> ParameterDescription<float>("label2", 11.0f) or
//                                  2 >> ParameterDescription<std::string>("label3", "aValue");
// It hold the temporary results of the operator>> and operator||
// functions in the expression.  The ONLY way to create one
// is via the operator>> function. The intent is that user
// should not need to save this temporary result nor directly
// reference this class, but ...

// If one decided to save the value then one should be aware
// the class has been optimized to minimize the number of copies made
// while evaluating such an expression.  It contains an auto_ptr and
// the class has copy semantics like an auto_ptr.  If you tried to use
// this class directly you must be aware that if a copy is
// made the original contains a null pointer. Then it would
// be easy to write code that dereferences that null pointer.

#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace edm {

  template<typename T>
  class ParameterDescriptionCases {
  public:
    typedef std::map<T, edm::value_ptr<ParameterDescriptionNode> > CaseMap;

    void insert(T caseValue, std::auto_ptr<ParameterDescriptionNode> node) {
      std::pair<T, edm::value_ptr<ParameterDescriptionNode> > casePair(caseValue,edm::value_ptr<ParameterDescriptionNode>());
      std::pair<typename CaseMap::iterator,bool> status;
      status = caseMap_->insert(casePair);
      (*caseMap_)[caseValue] = node;
      if (status.second == false) duplicateCaseValues_ = true;
    }

    std::auto_ptr<CaseMap> caseMap() { return caseMap_; }
    bool duplicateCaseValues() const { return duplicateCaseValues_; }

  private:

    friend
    std::auto_ptr<ParameterDescriptionCases<bool> >
    operator>>(bool caseValue,
               std::auto_ptr<ParameterDescriptionNode> node);

    friend
    std::auto_ptr<ParameterDescriptionCases<int> >
    operator>>(int caseValue,
               std::auto_ptr<ParameterDescriptionNode> node);

    friend
    std::auto_ptr<ParameterDescriptionCases<std::string> >
    operator>>(std::string const& caseValue,
               std::auto_ptr<ParameterDescriptionNode> node);

    friend
    std::auto_ptr<ParameterDescriptionCases<std::string> >
    operator>>(char const* caseValue,
               std::auto_ptr<ParameterDescriptionNode> node);

    // The constructor is intentionally private so that only the operator>> functions
    // can create these. 
    ParameterDescriptionCases(T const& caseValue, std::auto_ptr<ParameterDescriptionNode> node) :
      caseMap_(new CaseMap),
      duplicateCaseValues_(false)
    {
      std::pair<T, edm::value_ptr<ParameterDescriptionNode> > casePair(caseValue,edm::value_ptr<ParameterDescriptionNode>());
      caseMap_->insert(casePair);
      (*caseMap_)[caseValue] = node;
    }

    std::auto_ptr<CaseMap> caseMap_;
    bool duplicateCaseValues_;
  };

  std::auto_ptr<ParameterDescriptionCases<bool> >
  operator||(std::auto_ptr<ParameterDescriptionCases<bool> >,
             std::auto_ptr<ParameterDescriptionCases<bool> >);

  std::auto_ptr<ParameterDescriptionCases<int> >
  operator||(std::auto_ptr<ParameterDescriptionCases<int> >,
             std::auto_ptr<ParameterDescriptionCases<int> >);

  std::auto_ptr<ParameterDescriptionCases<std::string> >
  operator||(std::auto_ptr<ParameterDescriptionCases<std::string> >,
	     std::auto_ptr<ParameterDescriptionCases<std::string> >);
}
#endif
