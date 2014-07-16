#ifndef FWCore_ParameterSet_ParameterSwitch_h
#define FWCore_ParameterSet_ParameterSwitch_h

#include "FWCore/ParameterSet/interface/ParameterSwitchBase.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <sstream>
#include <ostream>
#include <iomanip>

namespace edm {

  template<class T>
  class ParameterSwitch : public ParameterSwitchBase {

  public:
    typedef std::map<T, edm::value_ptr<ParameterDescriptionNode> > CaseMap;
    typedef typename std::map<T, edm::value_ptr<ParameterDescriptionNode> >::const_iterator CaseMapConstIter;

    ParameterSwitch(ParameterDescription<T> const& switchParameter,
                    std::auto_ptr<ParameterDescriptionCases<T> > cases) :
      switch_(switchParameter),
      cases_(*cases->caseMap())
    {
      if (cases->duplicateCaseValues()) {
        throwDuplicateCaseValues(switchParameter.label());
      }
    }

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterSwitch(*this);
    }

  private:

    virtual void checkAndGetLabelsAndTypes_(std::set<std::string> & usedLabels,
                                            std::set<ParameterTypes> & parameterTypes,
                                            std::set<ParameterTypes> & wildcardTypes) const {

      std::set<std::string> caseLabels;
      std::set<ParameterTypes> caseParameterTypes;
      std::set<ParameterTypes> caseWildcardTypes;
      for_all(cases_, std::bind(&ParameterSwitch::checkCaseLabels,
                                  std::placeholders::_1,
                                  std::ref(caseLabels),
                                  std::ref(caseParameterTypes),
                                  std::ref(caseWildcardTypes)));

      insertAndCheckLabels(switch_.label(),
                           usedLabels,
                           caseLabels);

      insertAndCheckTypes(switch_.type(),
                          caseParameterTypes,
                          caseWildcardTypes,
                          parameterTypes,
                          wildcardTypes);

      if (cases_.find(switch_.getDefaultValue()) == cases_.end()) {
        throwNoCaseForDefault(switch_.label());
      }
    }

    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const {

      switch_.validate(pset, validatedLabels, optional);
      if (switch_.exists(pset)) {
        T switchValue;   
        if (switch_.isTracked()) {
          switchValue = pset.getParameter<T>(switch_.label());
        }
        else {
          switchValue = pset.getUntrackedParameter<T>(switch_.label());
        }
	typename CaseMap::const_iterator selectedCase = cases_.find(switchValue);
        if (selectedCase != cases_.end()) {
          selectedCase->second->validate(pset, validatedLabels, false);
        }
        else {
	  std::stringstream ss;
          ss << "The switch parameter with label \""
             << switch_.label()
             << "\" has been assigned an illegal value.\n"
             << "The value from the configuration is \""
             << switchValue
             << "\".\n"
             << "The allowed values are:\n";

          for (CaseMapConstIter iter = cases_.begin(), iEnd = cases_.end();
               iter != iEnd;
               ++iter) {
            ss << "  " << iter->first << "\n";
          }
          throwNoCaseForSwitchValue(ss.str());
        }
      }
    }

    virtual void writeCfi_(std::ostream & os,
                          bool & startWithComma,
                          int indentation,
                          bool & wroteSomething) const {
      switch_.writeCfi(os, startWithComma, indentation, wroteSomething);

      typename CaseMap::const_iterator selectedCase = cases_.find(switch_.getDefaultValue());
      if (selectedCase != cases_.end()) {
        selectedCase->second->writeCfi(os, startWithComma, indentation, wroteSomething);
      }
    }

    virtual void print_(std::ostream & os,
                        bool optional,
                        bool writeToCfi,
                        DocFormatHelper & dfh) {
      printBase(os, optional, writeToCfi, dfh, switch_.label(), switch_.isTracked(), parameterTypeEnumToString(switch_.type()));
    }

    virtual void printNestedContent_(std::ostream & os,
                                     bool optional,
                                     DocFormatHelper & dfh) {

      DocFormatHelper new_dfh(dfh);
      printNestedContentBase(os, dfh, new_dfh, switch_.label());

      switch_.print(os, optional, true, new_dfh);
      for_all(cases_, std::bind(&ParameterSwitchBase::printCaseT<T>,
                                  std::placeholders::_1,
                                  std::ref(os),
                                  optional,
                                  std::ref(new_dfh),
                                  std::cref(switch_.label())));

      new_dfh.setPass(1);
      new_dfh.setCounter(0);

      new_dfh.indent(os);
      os << "switch:\n";
      switch_.print(os, optional, true, new_dfh);
      for_all(cases_, std::bind(&ParameterSwitchBase::printCaseT<T>,
                                  std::placeholders::_1,
                                  std::ref(os),
                                  optional,
                                  std::ref(new_dfh),
                                  std::cref(switch_.label())));

      new_dfh.setPass(2);
      new_dfh.setCounter(0);

      switch_.printNestedContent(os, optional, new_dfh);
      for_all(cases_, std::bind(&ParameterSwitchBase::printCaseT<T>,
                                  std::placeholders::_1,
                                  std::ref(os),
                                  optional,
                                  std::ref(new_dfh),
                                  std::cref(switch_.label())));
    }

    virtual bool exists_(ParameterSet const& pset) const { return switch_.exists(pset); }

    static void checkCaseLabels(std::pair<T, edm::value_ptr<ParameterDescriptionNode> > const& thePair,
                                std::set<std::string> & labels,
                                std::set<ParameterTypes> & parameterTypes,
                                std::set<ParameterTypes> & wildcardTypes) {
      thePair.second->checkAndGetLabelsAndTypes(labels, parameterTypes, wildcardTypes);
    }

    ParameterDescription<T> switch_;
    CaseMap cases_;
  };
}
#endif
