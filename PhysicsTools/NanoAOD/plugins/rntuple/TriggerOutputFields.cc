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

#include <map>

using ROOT::RNTupleModel;

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
    const std::string& name, int index, const std::string& fieldName, const std::string& fieldDesc, RNTupleModel& model)
    : m_field(fieldName, fieldDesc, model), m_triggerName(name), m_triggerIndex(index) {}

void TriggerFieldPtr::fill(const edm::TriggerResults& triggers) {
  // A trigger absent from the current run's menu has no index and is filled as false
  m_field.fill(m_triggerIndex >= 0 ? triggers.accept(m_triggerIndex) : false);
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
    m_triggerFields.emplace_back(name, i, modelName, desc, model);
  }
}

void TriggerOutputFields::updateTriggerFields(const edm::TriggerResults& triggers) {
  std::vector<std::string> newNames(TriggerOutputFields::getTriggerNames(triggers));
  // Collect the current menu once. Each name is trimmed exactly once: trimVersionSuffix cuts at the
  // last "_v", so trimming an already trimmed name can truncate it again.
  std::map<std::string, int> menu;
  for (std::size_t j = 0; j < newNames.size(); j++) {
    auto& name = newNames[j];
    if (!isNanoaodTrigger(name)) {
      continue;
    }
    trimVersionSuffix(name);
    menu[name] = static_cast<int>(j);
  }
  // Point the existing fields at their index in the current menu, or at -1 if they are gone.
  // Erasing on a match leaves only the unclaimed paths behind for the warning below. It also means
  // a menu entry is bound to at most one field: should two fields ever carry the same trimmed name
  // (possible only if a menu holds two versions of one path), the first claims it and the rest are
  // written as false. That is the safe reading -- the alternative, pointing several fields at the
  // same bit, would silently duplicate one path's decision under several names.
  for (auto& t : m_triggerFields) {
    auto found = menu.find(t.getTriggerName());
    if (found == menu.end()) {
      t.setIndex(-1);
    } else {
      t.setIndex(found->second);
      menu.erase(found);
    }
  }
  // Whatever is left in the menu appeared after the schema was frozen and cannot be added.
  for (const auto& entry : menu) {
    // TODO backfill / friend ntuples
    edm::LogWarning("TriggerOutputFields") << "Skipping output of TriggerField " << entry.first << "\n";
  }
}

void TriggerOutputFields::makeUniqueFieldName(const RNTupleModel& model, std::string& name) {
  bool already_exists = model.GetFieldNames().contains(name);

  if (!already_exists) {
    return;
  }
  edm::LogWarning("TriggerOutputFields") << "Found a field with name " << name << " already present. Will add suffix _p"
                                         << m_processName << " to the new field.\n";
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
