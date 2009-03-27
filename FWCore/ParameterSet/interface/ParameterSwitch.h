#ifndef FWCore_ParameterSet_ParameterSwitch_h
#define FWCore_ParameterSet_ParameterSwitch_h

#include "FWCore/ParameterSet/interface/ParameterSwitchBase.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescriptionCases.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <iosfwd>
#include <sstream>

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
      for_all(cases_, boost::bind(&ParameterSwitch::checkCaseLabels,
                                  _1,
                                  boost::ref(caseLabels),
                                  boost::ref(caseParameterTypes),
                                  boost::ref(caseWildcardTypes)));

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
