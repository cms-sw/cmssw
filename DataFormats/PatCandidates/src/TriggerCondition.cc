//
//

#include "DataFormats/PatCandidates/interface/TriggerCondition.h"

using namespace pat;

// Constructors and Destructor

// Default constructor
TriggerCondition::TriggerCondition() : name_(), accept_(), category_(), type_() {
  triggerObjectTypes_.clear();
  objectKeys_.clear();
}

// Constructor from condition name "only"
TriggerCondition::TriggerCondition(const std::string& name) : name_(name), accept_(), category_(), type_() {
  triggerObjectTypes_.clear();
  objectKeys_.clear();
}

// Constructor from values
TriggerCondition::TriggerCondition(const std::string& name, bool accept)
    : name_(name), accept_(accept), category_(), type_() {
  triggerObjectTypes_.clear();
  objectKeys_.clear();
}

// Methods

// Get the trigger object types
std::vector<int> TriggerCondition::triggerObjectTypes() const {
  std::vector<int> triggerObjectTypes;
  for (auto triggerObjectType : triggerObjectTypes_) {
    triggerObjectTypes.push_back(int(triggerObjectType));
  }
  return triggerObjectTypes;
}

// Checks, if a certain trigger object type is assigned
bool TriggerCondition::hasTriggerObjectType(trigger::TriggerObjectType triggerObjectType) const {
  for (auto iT : triggerObjectTypes_) {
    if (iT == triggerObjectType)
      return true;
  }
  return false;
}

// Checks, if a certain trigger object collection index is assigned
bool TriggerCondition::hasObjectKey(unsigned objectKey) const {
  for (unsigned int iO : objectKeys_) {
    if (iO == objectKey)
      return true;
  }
  return false;
}
