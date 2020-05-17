//
//

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"

using namespace pat;

// Constructors and Destructor

// Default constructor
TriggerEvent::TriggerEvent()
    : nameL1Menu_(),
      nameHltTable_(),
      run_(),
      accept_(),
      error_(),
      physDecl_(),
      lhcFill_(),
      beamMode_(),
      beamMomentum_(),
      intensityBeam1_(),
      intensityBeam2_(),
      bstMasterStatus_(),
      turnCount_(),
      bCurrentStart_(),
      bCurrentStop_(),
      bCurrentAvg_() {
  objectMatchResults_.clear();
}

// Constructor from values, HLT only
TriggerEvent::TriggerEvent(const std::string& nameHltTable, bool run, bool accept, bool error, bool physDecl)
    : nameL1Menu_(),
      nameHltTable_(nameHltTable),
      run_(run),
      accept_(accept),
      error_(error),
      physDecl_(physDecl),
      lhcFill_(),
      beamMode_(),
      beamMomentum_(),
      intensityBeam1_(),
      intensityBeam2_(),
      bstMasterStatus_(),
      turnCount_(),
      bCurrentStart_(),
      bCurrentStop_(),
      bCurrentAvg_() {
  objectMatchResults_.clear();
}

// Constructor from values, HLT and L1/GT
TriggerEvent::TriggerEvent(
    const std::string& nameL1Menu, const std::string& nameHltTable, bool run, bool accept, bool error, bool physDecl)
    : nameL1Menu_(nameL1Menu),
      nameHltTable_(nameHltTable),
      run_(run),
      accept_(accept),
      error_(error),
      physDecl_(physDecl),
      lhcFill_(),
      beamMode_(),
      beamMomentum_(),
      intensityBeam1_(),
      intensityBeam2_(),
      bstMasterStatus_(),
      turnCount_(),
      bCurrentStart_(),
      bCurrentStop_(),
      bCurrentAvg_() {
  objectMatchResults_.clear();
}

// Methods

// Get a vector of references to all L1 algorithms
const TriggerAlgorithmRefVector TriggerEvent::algorithmRefs() const {
  TriggerAlgorithmRefVector theAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    const std::string nameAlgorithm(iAlgorithm.name());
    const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
    theAlgorithms.push_back(algorithmRef);
  }
  return theAlgorithms;
}

// Get a pointer to a certain L1 algorithm by name
const TriggerAlgorithm* TriggerEvent::algorithm(const std::string& nameAlgorithm) const {
  for (const auto& iAlgorithm : *algorithms()) {
    if (nameAlgorithm == iAlgorithm.name())
      return &iAlgorithm;
  }
  return nullptr;
}

// Get a reference to a certain L1 algorithm by name
const TriggerAlgorithmRef TriggerEvent::algorithmRef(const std::string& nameAlgorithm) const {
  for (auto&& iAlgorithm : algorithmRefs()) {
    if (nameAlgorithm == (iAlgorithm)->name())
      return iAlgorithm;
  }
  return TriggerAlgorithmRef();
}

// Get the name of a certain L1 algorithm in the event collection by bit number physics or technical algorithms,
std::string TriggerEvent::nameAlgorithm(const unsigned bitAlgorithm, const bool techAlgorithm) const {
  for (const auto& iAlgorithm : *algorithms()) {
    if (bitAlgorithm == iAlgorithm.bit() && techAlgorithm == iAlgorithm.techTrigger())
      return iAlgorithm.name();
  }
  return std::string("");
}

// Get the index of a certain L1 algorithm in the event collection by name
unsigned TriggerEvent::indexAlgorithm(const std::string& nameAlgorithm) const {
  unsigned iAlgorithm(0);
  while (iAlgorithm < algorithms()->size() && algorithms()->at(iAlgorithm).name() != nameAlgorithm)
    ++iAlgorithm;
  return iAlgorithm;
}

// Get a vector of references to all succeeding L1 algorithms
TriggerAlgorithmRefVector TriggerEvent::acceptedAlgorithms() const {
  TriggerAlgorithmRefVector theAcceptedAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (iAlgorithm.decision()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theAcceptedAlgorithms.push_back(algorithmRef);
    }
  }
  return theAcceptedAlgorithms;
}

// Get a vector of references to all L1 algorithms succeeding on the GTL board
TriggerAlgorithmRefVector TriggerEvent::acceptedAlgorithmsGtl() const {
  TriggerAlgorithmRefVector theAcceptedAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (iAlgorithm.gtlResult()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theAcceptedAlgorithms.push_back(algorithmRef);
    }
  }
  return theAcceptedAlgorithms;
}

// Get a vector of references to all technical L1 algorithms
TriggerAlgorithmRefVector TriggerEvent::techAlgorithms() const {
  TriggerAlgorithmRefVector theTechAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (iAlgorithm.techTrigger()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theTechAlgorithms.push_back(algorithmRef);
    }
  }
  return theTechAlgorithms;
}

// Get a vector of references to all succeeding technical L1 algorithms
TriggerAlgorithmRefVector TriggerEvent::acceptedTechAlgorithms() const {
  TriggerAlgorithmRefVector theAcceptedTechAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (iAlgorithm.techTrigger() && iAlgorithm.decision()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theAcceptedTechAlgorithms.push_back(algorithmRef);
    }
  }
  return theAcceptedTechAlgorithms;
}

// Get a vector of references to all technical L1 algorithms succeeding on the GTL board
TriggerAlgorithmRefVector TriggerEvent::acceptedTechAlgorithmsGtl() const {
  TriggerAlgorithmRefVector theAcceptedTechAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (iAlgorithm.techTrigger() && iAlgorithm.gtlResult()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theAcceptedTechAlgorithms.push_back(algorithmRef);
    }
  }
  return theAcceptedTechAlgorithms;
}

// Get a vector of references to all physics L1 algorithms
TriggerAlgorithmRefVector TriggerEvent::physAlgorithms() const {
  TriggerAlgorithmRefVector thePhysAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (!iAlgorithm.techTrigger()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      thePhysAlgorithms.push_back(algorithmRef);
    }
  }
  return thePhysAlgorithms;
}

// Get a vector of references to all succeeding physics L1 algorithms
TriggerAlgorithmRefVector TriggerEvent::acceptedPhysAlgorithms() const {
  TriggerAlgorithmRefVector theAcceptedPhysAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (!iAlgorithm.techTrigger() && iAlgorithm.decision()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theAcceptedPhysAlgorithms.push_back(algorithmRef);
    }
  }
  return theAcceptedPhysAlgorithms;
}

// Get a vector of references to all physics L1 algorithms succeeding on the GTL board
TriggerAlgorithmRefVector TriggerEvent::acceptedPhysAlgorithmsGtl() const {
  TriggerAlgorithmRefVector theAcceptedPhysAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    if (!iAlgorithm.techTrigger() && iAlgorithm.gtlResult()) {
      const std::string nameAlgorithm(iAlgorithm.name());
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theAcceptedPhysAlgorithms.push_back(algorithmRef);
    }
  }
  return theAcceptedPhysAlgorithms;
}

// Get a vector of references to all L1 conditions
const TriggerConditionRefVector TriggerEvent::conditionRefs() const {
  TriggerConditionRefVector theConditions;
  for (const auto& iCondition : *conditions()) {
    const std::string nameCondition(iCondition.name());
    const TriggerConditionRef conditionRef(conditions_, indexCondition(nameCondition));
    theConditions.push_back(conditionRef);
  }
  return theConditions;
}

// Get a pointer to a certain L1 condition by name
const TriggerCondition* TriggerEvent::condition(const std::string& nameCondition) const {
  for (const auto& iCondition : *conditions()) {
    if (nameCondition == iCondition.name())
      return &iCondition;
  }
  return nullptr;
}

// Get a reference to a certain L1 condition by name
const TriggerConditionRef TriggerEvent::conditionRef(const std::string& nameCondition) const {
  for (auto&& iCondition : conditionRefs()) {
    if (nameCondition == (iCondition)->name())
      return iCondition;
  }
  return TriggerConditionRef();
}

// Get the index of a certain L1 condition in the event collection by name
unsigned TriggerEvent::indexCondition(const std::string& nameCondition) const {
  unsigned iCondition(0);
  while (iCondition < conditions()->size() && conditions()->at(iCondition).name() != nameCondition)
    ++iCondition;
  return iCondition;
}

// Get a vector of references to all succeeding L1 conditions
TriggerConditionRefVector TriggerEvent::acceptedConditions() const {
  TriggerConditionRefVector theAcceptedConditions;
  for (const auto& iCondition : *conditions()) {
    if (iCondition.wasAccept()) {
      const std::string nameCondition(iCondition.name());
      const TriggerConditionRef conditionRef(conditions_, indexCondition(nameCondition));
      theAcceptedConditions.push_back(conditionRef);
    }
  }
  return theAcceptedConditions;
}

// Get a vector of references to all HLT paths
const TriggerPathRefVector TriggerEvent::pathRefs() const {
  TriggerPathRefVector thePaths;
  for (const auto& iPath : *paths()) {
    const std::string namePath(iPath.name());
    const TriggerPathRef pathRef(paths_, indexPath(namePath));
    thePaths.push_back(pathRef);
  }
  return thePaths;
}

// Get a pointer to a certain HLT path by name
const TriggerPath* TriggerEvent::path(const std::string& namePath) const {
  for (const auto& iPath : *paths()) {
    if (namePath == iPath.name())
      return &iPath;
  }
  return nullptr;
}

// Get a reference to a certain HLT path by name
const TriggerPathRef TriggerEvent::pathRef(const std::string& namePath) const {
  for (auto&& iPath : pathRefs()) {
    if (namePath == (iPath)->name())
      return iPath;
  }
  return TriggerPathRef();
}

// Get the index of a certain HLT path in the event collection by name
unsigned TriggerEvent::indexPath(const std::string& namePath) const {
  unsigned iPath(0);
  while (iPath < paths()->size() && paths()->at(iPath).name() != namePath)
    ++iPath;
  return iPath;
}

// Get a vector of references to all succeeding HLT paths
TriggerPathRefVector TriggerEvent::acceptedPaths() const {
  TriggerPathRefVector theAcceptedPaths;
  for (const auto& iPath : *paths()) {
    if (iPath.wasAccept()) {
      const std::string namePath(iPath.name());
      const TriggerPathRef pathRef(paths_, indexPath(namePath));
      theAcceptedPaths.push_back(pathRef);
    }
  }
  return theAcceptedPaths;
}

// Get a vector of references to all HLT filters
const TriggerFilterRefVector TriggerEvent::filterRefs() const {
  TriggerFilterRefVector theFilters;
  for (const auto& iFilter : *filters()) {
    const std::string labelFilter(iFilter.label());
    const TriggerFilterRef filterRef(filters_, indexFilter(labelFilter));
    theFilters.push_back(filterRef);
  }
  return theFilters;
}

// Get a pointer to a certain HLT filter by label
const TriggerFilter* TriggerEvent::filter(const std::string& labelFilter) const {
  for (const auto& iFilter : *filters()) {
    if (labelFilter == iFilter.label())
      return &iFilter;
  }
  return nullptr;
}

// Get a reference to a certain HLT filter by label
const TriggerFilterRef TriggerEvent::filterRef(const std::string& labelFilter) const {
  for (auto&& iFilter : filterRefs()) {
    if (labelFilter == (iFilter)->label())
      return iFilter;
  }
  return TriggerFilterRef();
}

// Get the index of a certain HLT filter in the event collection by label
unsigned TriggerEvent::indexFilter(const std::string& labelFilter) const {
  unsigned iFilter(0);
  while (iFilter < filters()->size() && filters()->at(iFilter).label() != labelFilter)
    ++iFilter;
  return iFilter;
}

// Get a vector of references to all succeeding HLT filters
TriggerFilterRefVector TriggerEvent::acceptedFilters() const {
  TriggerFilterRefVector theAcceptedFilters;
  for (const auto& iFilter : *filters()) {
    if (iFilter.status() == 1) {
      const std::string labelFilter(iFilter.label());
      const TriggerFilterRef filterRef(filters_, indexFilter(labelFilter));
      theAcceptedFilters.push_back(filterRef);
    }
  }
  return theAcceptedFilters;
}

// Get a vector of references to all trigger objects
const TriggerObjectRefVector TriggerEvent::objectRefs() const {
  TriggerObjectRefVector theObjects;
  for (unsigned iObject = 0; iObject < objects()->size(); ++iObject) {
    const TriggerObjectRef objectRef(objects_, iObject);
    theObjects.push_back(objectRef);
  }
  return theObjects;
}

// Get a vector of references to all trigger objects by trigger object type
TriggerObjectRefVector TriggerEvent::objects(trigger::TriggerObjectType triggerObjectType) const {
  TriggerObjectRefVector theObjects;
  for (unsigned iObject = 0; iObject < objects()->size(); ++iObject) {
    if (objects()->at(iObject).hasTriggerObjectType(triggerObjectType)) {
      const TriggerObjectRef objectRef(objects_, iObject);
      theObjects.push_back(objectRef);
    }
  }
  return theObjects;
}

// Get a vector of references to all conditions assigned to a certain algorithm given by name
TriggerConditionRefVector TriggerEvent::algorithmConditions(const std::string& nameAlgorithm) const {
  TriggerConditionRefVector theAlgorithmConditions;
  if (const TriggerAlgorithm* algorithmPtr = algorithm(nameAlgorithm)) {
    for (unsigned int iC : algorithmPtr->conditionKeys()) {
      const TriggerConditionRef conditionRef(conditions_, iC);
      theAlgorithmConditions.push_back(conditionRef);
    }
  }
  return theAlgorithmConditions;
}

// Checks, if a condition is assigned to a certain algorithm given by name
bool TriggerEvent::conditionInAlgorithm(const TriggerConditionRef& conditionRef,
                                        const std::string& nameAlgorithm) const {
  TriggerConditionRefVector theConditions = algorithmConditions(nameAlgorithm);
  for (auto&& theCondition : theConditions) {
    if (conditionRef == theCondition)
      return true;
  }
  return false;
}

// Get a vector of references to all algorithms, which have a certain condition assigned
TriggerAlgorithmRefVector TriggerEvent::conditionAlgorithms(const TriggerConditionRef& conditionRef) const {
  TriggerAlgorithmRefVector theConditionAlgorithms;
  size_t cAlgorithms(0);
  for (const auto& iAlgorithm : *algorithms()) {
    const std::string nameAlgorithm(iAlgorithm.name());
    if (conditionInAlgorithm(conditionRef, nameAlgorithm)) {
      const TriggerAlgorithmRef algorithmRef(algorithms_, cAlgorithms);
      theConditionAlgorithms.push_back(algorithmRef);
    }
    ++cAlgorithms;
  }
  return theConditionAlgorithms;
}

// Get a list of all trigger object collections used in a certain condition given by name
std::vector<std::string> TriggerEvent::conditionCollections(const std::string& nameCondition) const {
  std::vector<std::string> theConditionCollections;
  if (const TriggerCondition* conditionPtr = condition(nameCondition)) {
    for (unsigned iObject = 0; iObject < objects()->size(); ++iObject) {
      if (conditionPtr->hasObjectKey(iObject)) {
        bool found(false);
        std::string objectCollection(objects()->at(iObject).collection());
        for (const auto& theConditionCollection : theConditionCollections) {
          if (theConditionCollection == objectCollection) {
            found = true;
            break;
          }
        }
        if (!found) {
          theConditionCollections.push_back(objectCollection);
        }
      }
    }
  }
  return theConditionCollections;
}

// Get a vector of references to all objects, which were used in a certain condition given by name
TriggerObjectRefVector TriggerEvent::conditionObjects(const std::string& nameCondition) const {
  TriggerObjectRefVector theConditionObjects;
  if (const TriggerCondition* conditionPtr = condition(nameCondition)) {
    for (unsigned iObject = 0; iObject < objects()->size(); ++iObject) {
      if (conditionPtr->hasObjectKey(iObject)) {
        const TriggerObjectRef objectRef(objects_, iObject);
        theConditionObjects.push_back(objectRef);
      }
    }
  }
  return theConditionObjects;
}

// Checks, if an object was used in a certain condition given by name
bool TriggerEvent::objectInCondition(const TriggerObjectRef& objectRef, const std::string& nameCondition) const {
  if (const TriggerCondition* conditionPtr = condition(nameCondition))
    return conditionPtr->hasObjectKey(objectRef.key());
  return false;
}

// Get a vector of references to all conditions, which have a certain object assigned
TriggerConditionRefVector TriggerEvent::objectConditions(const TriggerObjectRef& objectRef) const {
  TriggerConditionRefVector theObjectConditions;
  for (const auto& iCondition : *conditions()) {
    const std::string nameCondition(iCondition.name());
    if (objectInCondition(objectRef, nameCondition)) {
      const TriggerConditionRef conditionRef(conditions_, indexCondition(nameCondition));
      theObjectConditions.push_back(conditionRef);
    }
  }
  return theObjectConditions;
}

// Get a vector of references to all objects, which were used in a certain algorithm given by name
TriggerObjectRefVector TriggerEvent::algorithmObjects(const std::string& nameAlgorithm) const {
  TriggerObjectRefVector theAlgorithmObjects;
  TriggerConditionRefVector theConditions = algorithmConditions(nameAlgorithm);
  for (auto&& theCondition : theConditions) {
    const std::string nameCondition((theCondition)->name());
    TriggerObjectRefVector theObjects = conditionObjects(nameCondition);
    for (auto&& theObject : theObjects) {
      theAlgorithmObjects.push_back(theObject);
    }
  }
  return theAlgorithmObjects;
}

// Checks, if an object was used in a certain algorithm given by name
bool TriggerEvent::objectInAlgorithm(const TriggerObjectRef& objectRef, const std::string& nameAlgorithm) const {
  TriggerConditionRefVector theConditions = algorithmConditions(nameAlgorithm);
  for (auto&& theCondition : theConditions) {
    if (objectInCondition(objectRef, (theCondition)->name()))
      return true;
  }
  return false;
}

// Get a vector of references to all algorithms, which have a certain object assigned
TriggerAlgorithmRefVector TriggerEvent::objectAlgorithms(const TriggerObjectRef& objectRef) const {
  TriggerAlgorithmRefVector theObjectAlgorithms;
  for (const auto& iAlgorithm : *algorithms()) {
    const std::string nameAlgorithm(iAlgorithm.name());
    if (objectInAlgorithm(objectRef, nameAlgorithm)) {
      const TriggerAlgorithmRef algorithmRef(algorithms_, indexAlgorithm(nameAlgorithm));
      theObjectAlgorithms.push_back(algorithmRef);
    }
  }
  return theObjectAlgorithms;
}

// Get a vector of references to all modules assigned to a certain path given by name
TriggerFilterRefVector TriggerEvent::pathModules(const std::string& namePath, bool all) const {
  TriggerFilterRefVector thePathFilters;
  if (const TriggerPath* pathPtr = path(namePath)) {
    if (!pathPtr->modules().empty()) {
      const unsigned onePastLastFilter = all ? pathPtr->modules().size() : pathPtr->lastActiveFilterSlot() + 1;
      for (unsigned iM = 0; iM < onePastLastFilter; ++iM) {
        const std::string labelFilter(pathPtr->modules().at(iM));
        const TriggerFilterRef filterRef(filters_, indexFilter(labelFilter));
        thePathFilters.push_back(filterRef);
      }
    }
  }
  return thePathFilters;
}

// Get a vector of references to all active HLT filters assigned to a certain path given by name
TriggerFilterRefVector TriggerEvent::pathFilters(const std::string& namePath, bool firing) const {
  TriggerFilterRefVector thePathFilters;
  if (const TriggerPath* pathPtr = path(namePath)) {
    for (unsigned int iF : pathPtr->filterIndices()) {
      const TriggerFilterRef filterRef(filters_, iF);
      if ((!firing) || filterRef->isFiring())
        thePathFilters.push_back(filterRef);
    }
  }
  return thePathFilters;
}

// Checks, if a filter is assigned to and was run in a certain path given by name
bool TriggerEvent::filterInPath(const TriggerFilterRef& filterRef, const std::string& namePath, bool firing) const {
  TriggerFilterRefVector theFilters = pathFilters(namePath, firing);
  for (auto&& theFilter : theFilters) {
    if (filterRef == theFilter)
      return true;
  }
  return false;
}

// Get a vector of references to all paths, which have a certain filter assigned
TriggerPathRefVector TriggerEvent::filterPaths(const TriggerFilterRef& filterRef, bool firing) const {
  TriggerPathRefVector theFilterPaths;
  size_t cPaths(0);
  for (const auto& iPath : *paths()) {
    const std::string namePath(iPath.name());
    if (filterInPath(filterRef, namePath, firing)) {
      const TriggerPathRef pathRef(paths_, cPaths);
      theFilterPaths.push_back(pathRef);
    }
    ++cPaths;
  }
  return theFilterPaths;
}

// Get a list of all trigger object collections used in a certain filter given by name
std::vector<std::string> TriggerEvent::filterCollections(const std::string& labelFilter) const {
  std::vector<std::string> theFilterCollections;
  if (const TriggerFilter* filterPtr = filter(labelFilter)) {
    for (unsigned iObject = 0; iObject < objects()->size(); ++iObject) {
      if (filterPtr->hasObjectKey(iObject)) {
        bool found(false);
        const std::string objectCollection(objects()->at(iObject).collection());
        for (const auto& theFilterCollection : theFilterCollections) {
          if (theFilterCollection == objectCollection) {
            found = true;
            break;
          }
        }
        if (!found) {
          theFilterCollections.push_back(objectCollection);
        }
      }
    }
  }
  return theFilterCollections;
}

// Get a vector of references to all objects, which were used in a certain filter given by name
TriggerObjectRefVector TriggerEvent::filterObjects(const std::string& labelFilter) const {
  TriggerObjectRefVector theFilterObjects;
  if (const TriggerFilter* filterPtr = filter(labelFilter)) {
    for (unsigned iObject = 0; iObject < objects()->size(); ++iObject) {
      if (filterPtr->hasObjectKey(iObject)) {
        const TriggerObjectRef objectRef(objects_, iObject);
        theFilterObjects.push_back(objectRef);
      }
    }
  }
  return theFilterObjects;
}

// Checks, if an object was used in a certain filter given by name
bool TriggerEvent::objectInFilter(const TriggerObjectRef& objectRef, const std::string& labelFilter) const {
  if (const TriggerFilter* filterPtr = filter(labelFilter))
    return filterPtr->hasObjectKey(objectRef.key());
  return false;
}

// Get a vector of references to all filters, which have a certain object assigned
TriggerFilterRefVector TriggerEvent::objectFilters(const TriggerObjectRef& objectRef, bool firing) const {
  TriggerFilterRefVector theObjectFilters;
  for (const auto& iFilter : *filters()) {
    const std::string labelFilter(iFilter.label());
    if (objectInFilter(objectRef, labelFilter)) {
      const TriggerFilterRef filterRef(filters_, indexFilter(labelFilter));
      if ((!firing) || iFilter.isFiring())
        theObjectFilters.push_back(filterRef);
    }
  }
  return theObjectFilters;
}

// Get a vector of references to all objects, which were used in a certain path given by name
TriggerObjectRefVector TriggerEvent::pathObjects(const std::string& namePath, bool firing) const {
  TriggerObjectRefVector thePathObjects;
  TriggerFilterRefVector theFilters = pathFilters(namePath, firing);
  for (auto&& theFilter : theFilters) {
    const std::string labelFilter((theFilter)->label());
    TriggerObjectRefVector theObjects = filterObjects(labelFilter);
    for (auto&& theObject : theObjects) {
      thePathObjects.push_back(theObject);
    }
  }
  return thePathObjects;
}

// Checks, if an object was used in a certain path given by name
bool TriggerEvent::objectInPath(const TriggerObjectRef& objectRef, const std::string& namePath, bool firing) const {
  TriggerFilterRefVector theFilters = pathFilters(namePath, firing);
  for (auto&& theFilter : theFilters) {
    if (objectInFilter(objectRef, (theFilter)->label()))
      return true;
  }
  return false;
}

// Get a vector of references to all paths, which have a certain object assigned
TriggerPathRefVector TriggerEvent::objectPaths(const TriggerObjectRef& objectRef, bool firing) const {
  TriggerPathRefVector theObjectPaths;
  for (const auto& iPath : *paths()) {
    const std::string namePath(iPath.name());
    if (objectInPath(objectRef, namePath, firing)) {
      const TriggerPathRef pathRef(paths_, indexPath(namePath));
      theObjectPaths.push_back(pathRef);
    }
  }
  return theObjectPaths;
}

// Add a pat::TriggerObjectMatch association
bool TriggerEvent::addObjectMatchResult(const TriggerObjectMatchRefProd& trigMatches, const std::string& labelMatcher) {
  if (triggerObjectMatchResults()->find(labelMatcher) == triggerObjectMatchResults()->end()) {
    objectMatchResults_[labelMatcher] = trigMatches;
    return true;
  }
  return false;
}

// Get a list of all linked trigger matches
std::vector<std::string> TriggerEvent::triggerMatchers() const {
  std::vector<std::string> theMatchers;
  for (const auto& iMatch : *triggerObjectMatchResults())
    theMatchers.push_back(iMatch.first);
  return theMatchers;
}

// Get a pointer to a certain trigger match given by label
const TriggerObjectMatch* TriggerEvent::triggerObjectMatchResult(const std::string& labelMatcher) const {
  const auto iMatch(triggerObjectMatchResults()->find(labelMatcher));
  if (iMatch != triggerObjectMatchResults()->end())
    return iMatch->second.get();
  return nullptr;
}
