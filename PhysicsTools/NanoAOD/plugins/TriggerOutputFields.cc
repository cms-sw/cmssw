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

namespace {

void trim_version_suffix(std::string& trigger_name) {
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

bool is_nanoaod_trigger(const std::string& name) {
  return name.compare(0, 3, "HLT") == 0 || name.compare(0, 4, "Flag") == 0
    || name.compare(0, 2, "L1") == 0;
}

} // anonymous namespace

std::vector<std::string> TriggerOutputFields::getTriggerNames(
  const edm::TriggerResults& triggerResults)
{
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
  // TODO check if handle is valid?
  const edm::TriggerResults& triggerResults = *handle;
  std::vector<std::string> triggerNames(TriggerOutputFields::getTriggerNames(triggerResults));
  m_triggerFields.reserve(triggerNames.size());
  for (std::size_t i = 0; i < triggerNames.size(); i++) {
    auto& name = triggerNames[i];
    if (!is_nanoaod_trigger(name)) {
      continue;
    }
    trim_version_suffix(name);
    makeUniqueFieldName(model, name);
    m_triggerFields.emplace_back(RNTupleFieldPtr<bool>(name, model));
    m_triggerFieldIndices.push_back(i);
  }
}

void TriggerOutputFields::makeUniqueFieldName(RNTupleModel& model, std::string& name) {
  // fix with cache of names in a higher-level object, don't ask the RNTupleModel each time
  const auto* existing_field = model.Get<bool>(name);
  if (!existing_field) {
    return;
  }
  edm::LogWarning("TriggerOutputFields") << "Found a branch with name " << name
    << " already present. Will add suffix _p" << m_processName << " to the new branch.\n";
  name += std::string("_p") + m_processName;
}

void TriggerOutputFields::fill(const edm::EventForOutput& event) {
  if (m_lastRun != event.id().run()) {
    std::cout << "skipping trigger output from run " << event.id().run() << "\n";
    return;
  }
  edm::Handle<edm::TriggerResults> handle;
  event.getByToken(m_token, handle);
  const edm::TriggerResults& triggers = *handle;
  for (std::size_t i = 0; i < m_triggerFields.size(); i++) {
    m_triggerFields[i].fill(triggers.accept(m_triggerFieldIndices.at(i)));
  }
}
