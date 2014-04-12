
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

namespace edm {

  TriggerResultsByName::
  TriggerResultsByName(TriggerResults const* triggerResults,
                       TriggerNames const* triggerNames) :
    triggerResults_(triggerResults),
    triggerNames_(triggerNames) {

   // If either of these is true the object is in an invalid state
   if (triggerResults_ == 0 || triggerNames_ == 0) {
     return;
   }

   if (triggerResults_->parameterSetID() != triggerNames_->parameterSetID()) {
      throw edm::Exception(edm::errors::Unknown)
        << "TriggerResultsByName::TriggerResultsByName, Trigger names vector and TriggerResults\n"
           "have different ParameterSetID's.  This should be impossible when the object is obtained\n"
           "from the function named triggerResultsByName in the Event class, which is the way\n"
           "TriggerResultsByName should always be created. If this is the case, please send\n"
           "information to reproduce this problem to the edm developers.  Otherwise, you are\n"
	   "using this class incorrectly and in a way that is not supported.\n";
    }

    if (triggerResults_->size() != triggerNames_->size()) {
      throw edm::Exception(edm::errors::Unknown)
        << "TriggerResultsByName::TriggerResultsByName, Trigger names vector\n"
           "and TriggerResults have different sizes.  This should be impossible,\n"
           "please send information to reproduce this problem to the edm developers.\n";
    }
  }

  bool
  TriggerResultsByName::
  isValid() const {
    if (triggerResults_ == 0 || triggerNames_ == 0) return false;
    return true;
  }

  ParameterSetID const&
  TriggerResultsByName::
  parameterSetID() const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->parameterSetID();
  }

  bool
  TriggerResultsByName::
  wasrun() const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->wasrun();
  }

  bool
  TriggerResultsByName::
  accept() const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->accept();
  }

  bool
  TriggerResultsByName::
  error() const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->error();
  }

  HLTPathStatus const& 
  TriggerResultsByName::
  at(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->at(i);
  }

  HLTPathStatus const&
  TriggerResultsByName::
  at(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->at(i);
  }

  HLTPathStatus const& 
  TriggerResultsByName::
  operator[](std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->at(i);
  }

  HLTPathStatus const&
  TriggerResultsByName::
  operator[](unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->at(i);
  }

  bool
  TriggerResultsByName::
  wasrun(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->wasrun(i);
  }

  bool
  TriggerResultsByName::
  wasrun(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->wasrun(i);
  }

  bool
  TriggerResultsByName::
  accept(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->accept(i);
  }

  bool
  TriggerResultsByName::
  accept(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->accept(i);
  }

  bool
  TriggerResultsByName::
  error(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->error(i);
  }

  bool
  TriggerResultsByName::
  error(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->error(i);
  }

  hlt::HLTState
  TriggerResultsByName::
  state(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->state(i);
  }

  hlt::HLTState
  TriggerResultsByName::
  state(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->state(i);
  }

  unsigned
  TriggerResultsByName::
  index(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    unsigned i = getAndCheckIndex(pathName);
    return triggerResults_->index(i);
  }

  unsigned
  TriggerResultsByName::
  index(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->index(i);
  }

  std::vector<std::string> const&
  TriggerResultsByName::
  triggerNames() const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    if (triggerNames_ == 0) throwTriggerNamesMissing(); 
    return triggerNames_->triggerNames();
  }

  std::string const&
  TriggerResultsByName::
  triggerName(unsigned i) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    if (triggerNames_ == 0) throwTriggerNamesMissing(); 
    return triggerNames_->triggerName(i);    
  }

  unsigned
  TriggerResultsByName::
  triggerIndex(std::string const& pathName) const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    if (triggerNames_ == 0) throwTriggerNamesMissing(); 
    return triggerNames_->triggerIndex(pathName);    
  }

  std::vector<std::string>::size_type
  TriggerResultsByName::
  size() const {
    if (triggerResults_ == 0) throwTriggerResultsMissing();
    return triggerResults_->size();
  }

  unsigned
  TriggerResultsByName::
  getAndCheckIndex(std::string const& pathName) const {
    if (triggerNames_ == 0) throwTriggerNamesMissing(); 
    unsigned i = triggerNames_->triggerIndex(pathName);
    if (i == triggerNames_->size()) {
      throw edm::Exception(edm::errors::LogicError)
        << "TriggerResultsByName::getAndCheckIndex\n"
	<< "Requested trigger name \""
        << pathName << "\" does not match any known trigger.\n";
    }
    return i;
  }

  void
  TriggerResultsByName::
  throwTriggerResultsMissing() const {
    throw edm::Exception(edm::errors::ProductNotFound)
      << "TriggerResultsByName has a null pointer to TriggerResults.\n"
      << "This probably means TriggerResults was not found in the Event\n"
      << "because the product was dropped or never created. It could also\n"
      << "mean it was requested for a process that does not exist or was\n"
      << "misspelled\n";
  }

  void
  TriggerResultsByName::
  throwTriggerNamesMissing() const {
    throw edm::Exception(edm::errors::LogicError)
      << "TriggerResultsByName has a null pointer to TriggerNames.\n"
      << "This should never happen. It could indicate the ParameterSet\n"
      << "containing the names is missing from the ParameterSet registry.\n"
      << "Please report this to the edm developers along with instructions\n"
      << "to reproduce the problem\n";
  }
}
