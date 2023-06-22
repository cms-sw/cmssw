#include "TriggerOutputFields.h"

#include "RNTupleFieldPtr.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

namespace {

  void trimVersionSuffix(std::string& trigger_name) {
    // HLT and L1 triggers have version suffixes we trim before filling the RNTuple
    if (trigger_name.compare(0, 3, "HLT") != 0 && trigger_name.compare(0, 2, "L1") != 0) {
      return;
    }
    auto vfound = trigger_name.rfind("_v");
    if (vfound == std::string::npos) {
      return;
    }
    trigger_name.replace(vfound, trigger_name.size() - vfound, "");
  }

  bool isNanoaodTrigger(const std::string& name) {
    return name.compare(0, 3, "HLT") == 0 || name.compare(0, 4, "Flag") == 0 || name.compare(0, 2, "L1") == 0;
  }

}  // anonymous namespace

TriggerFieldPtr::TriggerFieldPtr(
    std::string name, int index, std::string fieldName, std::string fieldDesc, RNTupleModel& model)
    : m_triggerName(name), m_triggerIndex(index) {
  m_field = RNTupleFieldPtr<bool>(fieldName, fieldDesc, model);
}

void TriggerFieldPtr::fill(const edm::TriggerResults& triggers) {
  if (m_triggerIndex == -1) {
    m_field.fill(false);
  }
  m_field.fill(triggers.accept(m_triggerIndex));
}

std::vector<std::string> TriggerOutputFields::getTriggerNames(const edm::TriggerResults& triggerResults) {
  // Trigger names are either stored in the TriggerResults object (e.g. L1) or
  // need to be looked up in the registry (e.g. HLT)
  auto triggerNames = triggerResults.getTriggerNames();
  if (!triggerNames.empty()) {
    return triggerNames;
  }
  edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
  edm::ParameterSet const* pset = psetRegistry->getMapped(triggerResults.parameterSetID());
  if (nullptr == pset || !pset->existsAs<std::vector<std::string>>("@trigger_paths", true)) {
    return {};
  }
  edm::TriggerNames names(*pset);
  if (names.size() != triggerResults.size()) {
    throw cms::Exception("LogicError") << "TriggerOutputFields::getTriggerNames "
                                          "Encountered vector\n of trigger names and a TriggerResults object with\n"
                                          "different sizes.  This should be impossible.\n"
                                          "Please send information to reproduce this problem to\nthe edm developers.\n";
  }
  return names.triggerNames();
}

void TriggerOutputFields::createFields(const edm::EventForOutput& event, RNTupleModel& model) {
  m_lastRun = event.id().run();
  edm::Handle<edm::TriggerResults> handle;
  event.getByToken(m_token, handle);
  const edm::TriggerResults& triggerResults = *handle;
  std::vector<std::string> triggerNames(TriggerOutputFields::getTriggerNames(triggerResults));
  m_triggerFields.reserve(triggerNames.size());
  for (std::size_t i = 0; i < triggerNames.size(); i++) {
    auto& name = triggerNames[i];
    if (!isNanoaodTrigger(name)) {
      continue;
    }
    trimVersionSuffix(name);
    std::string modelName = name;
    makeUniqueFieldName(model, modelName);
    std::string desc = std::string("Trigger/flag bit (process: ") + m_processName + ")";
    m_triggerFields.emplace_back(TriggerFieldPtr(name, i, modelName, desc, model));
  }
}

// Worst case O(n^2) to adjust the triggers
void TriggerOutputFields::updateTriggerFields(const edm::TriggerResults& triggers) {
  std::vector<std::string> newNames(TriggerOutputFields::getTriggerNames(triggers));
  // adjust existing trigger indices
  for (auto& t : m_triggerFields) {
    t.setIndex(-1);
    for (std::size_t j = 0; j < newNames.size(); j++) {
      auto& name = newNames[j];
      if (!isNanoaodTrigger(name)) {
        continue;
      }
      trimVersionSuffix(name);
      if (name == t.getTriggerName()) {
        t.setIndex(j);
      }
    }
  }
  // find new triggers
  for (std::size_t j = 0; j < newNames.size(); j++) {
    auto& name = newNames[j];
    if (!isNanoaodTrigger(name)) {
      continue;
    }
    trimVersionSuffix(name);
    if (std::none_of(m_triggerFields.cbegin(), m_triggerFields.cend(), [&](const TriggerFieldPtr& t) {
          return t.getTriggerName() == name;
        })) {
      // TODO backfill / friend ntuples
      edm::LogWarning("TriggerOutputFields") << "Skipping output of TriggerField " << name << "\n";
    }
  }
}

void TriggerOutputFields::makeUniqueFieldName(RNTupleModel& model, std::string& name) {
  // Could also use a cache of names in a higher-level object, don't ask the RNTupleModel each time
  const auto* existing_field = model.Get<bool>(name);
  if (!existing_field) {
    return;
  }
  edm::LogWarning("TriggerOutputFields") << "Found a branch with name " << name
                                         << " already present. Will add suffix _p" << m_processName
                                         << " to the new branch.\n";
  name += std::string("_p") + m_processName;
}

void TriggerOutputFields::fill(const edm::EventForOutput& event) {
  edm::Handle<edm::TriggerResults> handle;
  event.getByToken(m_token, handle);
  const edm::TriggerResults& triggers = *handle;
  if (m_lastRun != event.id().run()) {
    m_lastRun = event.id().run();
    updateTriggerFields(triggers);
  }
  for (auto& t : m_triggerFields) {
    t.fill(triggers);
  }
}
