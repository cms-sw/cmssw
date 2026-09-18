#include "TriggerOutputFields.h"

#include "RNTupleProjections.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <ROOT/RField.hxx>
#include <ROOT/RField/RFieldRecord.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RVersion.hxx>

#include <algorithm>
#include <array>
#include <cctype>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>

using ROOT::REntry;
using ROOT::RFieldBase;
using ROOT::RNTupleModel;
using ROOT::RNTupleWriter;
using ROOT::RRecordField;

namespace {

  // The prefixes NanoAOD keeps, one record each. No prefix here may be a prefix of another.
  const std::array<std::string, 3> kTriggerGroups = {"HLT", "Flag", "L1"};

  // A version suffix is "_v" followed by the version number and nothing else. "_v" with an empty
  // tail counts: unversioned menu entries are written that way, and the TTree module trims them too.
  bool isVersionSuffix(const std::string& name, std::size_t vpos) {
    for (std::size_t i = vpos + 2; i < name.size(); i++) {
      if (std::isdigit(static_cast<unsigned char>(name[i])) == 0) {
        return false;
      }
    }
    return true;
  }

  void trimVersionSuffix(std::string& trigger_name) {
    // HLT and L1 triggers have version suffixes we trim before filling the RNTuple
    if (trigger_name.compare(0, 3, "HLT") != 0 && trigger_name.compare(0, 2, "L1") != 0) {
      return;
    }
    // Only cut a real version. Cutting at the last "_v" whatever follows it would also eat the tail
    // of a path that merely contains one -- an "HLT_DiJet_vbf" would land on "HLT_DiJet" and could
    // then collide with a genuine "HLT_DiJet_v3", costing one of the two its own field.
    auto vfound = trigger_name.rfind("_v");
    if (vfound == std::string::npos || !isVersionSuffix(trigger_name, vfound)) {
      return;
    }
    trigger_name.replace(vfound, trigger_name.size() - vfound, "");
  }

  // Where a path goes: its group's record, the member name inside it, and the flat name the TTree
  // module gives the branch, which is the path name with any version suffix trimmed.
  struct SplitTriggerName {
    std::string group;
    std::string member;
    std::string flatName;
  };

  // The record and member a path belongs in, or nothing if NanoAOD does not keep it.
  // HLT_IsoMu24_v3 is ("HLT", "IsoMu24"), so it is written as HLT.IsoMu24. A kept path that is not
  // of the form <group>_<rest>, such as HLTriggerFinalPath, keeps its whole name as the member --
  // the one case where the flat name is not the group and the member joined by "_" again.
  std::optional<SplitTriggerName> splitTriggerName(std::string name) {
    const auto group = std::find_if(kTriggerGroups.begin(), kTriggerGroups.end(), [&name](const std::string& prefix) {
      return name.compare(0, prefix.size(), prefix) == 0;
    });
    if (group == kTriggerGroups.end()) {
      return std::nullopt;
    }
    trimVersionSuffix(name);
    if (name.size() > group->size() + 1 && name[group->size()] == '_') {
      return SplitTriggerName{*group, name.substr(group->size() + 1), name};
    }
    return SplitTriggerName{*group, name, name};
  }

  std::string memberDescription(const std::string& processName) {
    return "Trigger/flag bit (process: " + processName + ")";
  }

}  // anonymous namespace

TriggerRecordFields::TriggerRecordFields(const std::string& groupName, const std::string& processName)
    : m_groupName(groupName), m_fieldName(groupName), m_processName(processName) {}

void TriggerRecordFields::addPath(const std::string& member, const TriggerMenuEntry& path) {
  auto found = m_positions.find(member);
  if (found != m_positions.end()) {
    // Two paths of one menu trimming to the same name: keep a single member and follow the last of
    // them, which is what update() settles on too, its menu being keyed on the trimmed name.
    edm::LogWarning("TriggerOutputFields")
        << "More than one path in the " << m_processName << " menu maps to " << m_groupName << "." << member
        << "; writing a single field, following the last of them.\n";
    m_indices[found->second] = path.index;
    return;
  }
  m_positions.emplace(member, m_members.size());
  m_members.push_back(member);
  m_flatNames.push_back(path.flatName);
  m_indices.push_back(path.index);
}

void TriggerRecordFields::createField(RNTupleModel& model, const std::string& fieldName) {
  m_fieldName = fieldName;
  std::vector<std::unique_ptr<RFieldBase>> subfields;
  subfields.reserve(m_members.size());
  for (const auto& member : m_members) {
    auto field = RFieldBase::Create(member, "bool").Unwrap();
    field->SetDescription(memberDescription(m_processName));
    subfields.push_back(std::move(field));
  }
  auto record = std::make_unique<RRecordField>(m_fieldName, std::move(subfields));
  record->SetDescription("Trigger/flag bits (process: " + m_processName + ")");
  model.AddField(std::move(record));
}

void TriggerRecordFields::addProjections(RNTupleModel& model) const {
  // Not rntupleprojection::addForField: a path's flat name is not always the record and the member
  // joined by "_", so each member is named here rather than derived from the record.
  for (std::size_t i = 0; i < m_members.size(); i++) {
    rntupleprojection::add(
        model, m_flatNames[i], m_fieldName, m_members[i], "bool", memberDescription(m_processName), false);
  }
}

void TriggerRecordFields::bind(REntry& entry, const RNTupleModel& model) {
  const auto* record = dynamic_cast<const RRecordField*>(&model.GetConstField(m_fieldName));
  if (nullptr == record) {
    throw cms::Exception("LogicError", "Trigger field " + m_fieldName + " is not a record");
  }
  m_offsets = record->GetOffsets();
  if (m_offsets.size() != m_members.size()) {
    throw cms::Exception("LogicError",
                         "Trigger record " + m_fieldName + " has " + std::to_string(m_offsets.size()) +
                             " members in the model but " + std::to_string(m_members.size()) + " here");
  }
  m_buffer.assign(record->GetValueSize(), 0);
  entry.BindRawPtr<void>(m_fieldName, m_buffer.data());
  for (std::size_t i = 0; i < m_looseFlatNames.size(); i++) {
    entry.BindValue<bool>(m_looseFlatNames[i], m_looseValues[i]);
  }
}

bool TriggerRecordFields::update(TriggerMenu& menu, RNTupleWriter& writer, [[maybe_unused]] bool addProjections) {
  // Point the paths already written at their index in the current menu, or at -1 if they are gone.
  // Erasing on a match leaves only the paths this record does not have yet. A path that came back
  // under a different flat name -- a version suffix appearing or going -- keeps the name it was
  // written under; renaming a field mid-ntuple is not possible, and the member name is the identity.
  for (std::size_t i = 0; i < m_members.size(); i++) {
    auto found = menu.find(m_members[i]);
    if (found == menu.end()) {
      m_indices[i] = -1;
    } else {
      m_indices[i] = found->second.index;
      menu.erase(found);
    }
  }
  for (std::size_t i = 0; i < m_looseMembers.size(); i++) {
    auto found = menu.find(m_looseMembers[i]);
    if (found == menu.end()) {
      m_looseIndices[i] = -1;
    } else {
      m_looseIndices[i] = found->second.index;
      menu.erase(found);
    }
  }
  if (menu.empty()) {
    return false;
  }
  // Every top-level name the model holds. RNTuple throws on a duplicate rather than reporting it,
  // so a name has to be checked before it is used; taken here, before the model starts changing.
  std::unordered_set<std::string> taken = writer.GetModel().GetFieldNames();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)
  {
    // The updater commits when it goes out of scope. Committing explicitly as well commits twice,
    // which leaves the model id at 0 and makes the next Fill() throw "mismatch between entry and
    // model"; the updater must also not outlive the writer, whose model its destructor re-freezes.
    auto updater = writer.CreateModelUpdater();
    updater->BeginUpdate();
    for (const auto& path : menu) {
      auto field = RFieldBase::Create(path.first, "bool").Unwrap();
      field->SetDescription(memberDescription(m_processName));
      updater->AddField(std::move(field), m_fieldName);
      m_positions.emplace(path.first, m_members.size());
      m_members.push_back(path.first);
      m_flatNames.push_back(path.second.flatName);
      m_indices.push_back(path.second.index);
      if (addProjections) {
        // The projection resolves against the member added just above, in the same transaction.
        rntupleprojection::add(*updater,
                               taken,
                               path.second.flatName,
                               m_fieldName,
                               path.first,
                               "bool",
                               memberDescription(m_processName),
                               false);
      }
    }
  }
  for (const auto& path : menu) {
    edm::LogInfo("TriggerOutputFields") << "Added " << m_fieldName << "." << path.first
                                        << ", which is not in the menu the schema was built from; it reads false for "
                                           "the entries written before now.\n";
  }
  menu.clear();
  return true;
#else
  // Before 6.40 a record cannot grow once the ntuple is being written. The path still gets into the
  // file, as a top-level bool field under the flat name the TTree module gives its branch, which is
  // also the name a projection of the member would have had. It is written whether or not
  // projections are on: the alternative is dropping the path.
  //
  // Settle on the names first: opening an updater invalidates the entry the caller is filling
  // through, so it must only be opened when there is something to add.
  std::vector<std::pair<std::string, TriggerMenuEntry>> toAdd;
  std::vector<std::string> added;
  for (const auto& path : menu) {
    if (!taken.insert(path.second.flatName).second) {
      edm::LogWarning("TriggerOutputFields")
          << "Skipping output of " << m_fieldName << "." << path.first << ": it is not in the menu the schema was "
          << "built from, so it would have to be written as " << path.second.flatName
          << ", and a field of that name is already there.\n";
      continue;
    }
    toAdd.push_back(path);
    added.push_back(path.second.flatName);
  }
  if (!toAdd.empty()) {
    auto updater = writer.CreateModelUpdater();
    updater->BeginUpdate();
    for (const auto& path : toAdd) {
      auto field = RFieldBase::Create(path.second.flatName, "bool").Unwrap();
      field->SetDescription(memberDescription(m_processName));
      updater->AddField(std::move(field));
      m_looseMembers.push_back(path.first);
      m_looseFlatNames.push_back(path.second.flatName);
      m_looseIndices.push_back(path.second.index);
      m_looseValues.push_back(std::make_shared<bool>(false));
    }
  }
  for (const auto& name : added) {
    edm::LogWarning("TriggerOutputFields")
        << "Writing " << name << " as a top-level field rather than a member of " << m_fieldName
        << ": it is not in the menu the schema was built from, and growing the record needs ROOT 6.40. It reads false "
           "for the entries written before now.\n";
  }
  menu.clear();
  return !added.empty();
#endif
}

void TriggerRecordFields::fill(const edm::TriggerResults& triggers) {
  for (std::size_t i = 0; i < m_indices.size(); i++) {
    // A path absent from the current run's menu has no index and is filled as false
    m_buffer[m_offsets[i]] = (m_indices[i] >= 0 && triggers.accept(m_indices[i])) ? 1 : 0;
  }
  for (std::size_t i = 0; i < m_looseIndices.size(); i++) {
    *m_looseValues[i] = m_looseIndices[i] >= 0 && triggers.accept(m_looseIndices[i]);
  }
}

///////////////////////////////////////////////////////////////////////////////

void TriggerOutputFields::createFields(const edm::EventForOutput& event, RNTupleModel& model) {
  m_lastRun = event.id().run();
  edm::Handle<edm::TriggerResults> handle;
  event.getByToken(m_token, handle);
  std::vector<std::string> triggerNames(TriggerOutputFields::getTriggerNames(*handle));
  for (std::size_t i = 0; i < triggerNames.size(); i++) {
    auto split = splitTriggerName(triggerNames[i]);
    if (!split) {
      continue;
    }
    auto record = std::find_if(m_records.begin(), m_records.end(), [&split](const TriggerRecordFields& candidate) {
      return candidate.getGroupName() == split->group;
    });
    if (record == m_records.end()) {
      record = m_records.emplace(m_records.end(), split->group, m_processName);
    }
    record->addPath(split->member, {static_cast<int>(i), split->flatName});
  }
  // Named only now, when the whole model is there to be checked against for a name clash.
  for (auto& record : m_records) {
    record.createField(model, uniqueName(model, record.getGroupName()));
  }
}

void TriggerOutputFields::addProjections(RNTupleModel& model) const {
  for (const auto& record : m_records) {
    record.addProjections(model);
  }
}

void TriggerOutputFields::bind(REntry& entry, const RNTupleModel& model) {
  for (auto& record : m_records) {
    record.bind(entry, model);
  }
}

bool TriggerOutputFields::updateForRun(const edm::EventForOutput& event, RNTupleWriter& writer, bool addProjections) {
  if (m_lastRun == static_cast<long>(event.id().run())) {
    return false;
  }
  m_lastRun = event.id().run();
  edm::Handle<edm::TriggerResults> handle;
  event.getByToken(m_token, handle);
  // Collect the current menu once. Each name is trimmed exactly once: trimVersionSuffix cuts at the
  // last "_v", so trimming an already trimmed name can truncate it again.
  std::map<std::string, TriggerMenu> menu;
  std::vector<std::string> triggerNames(TriggerOutputFields::getTriggerNames(*handle));
  for (std::size_t i = 0; i < triggerNames.size(); i++) {
    auto split = splitTriggerName(triggerNames[i]);
    if (!split) {
      continue;
    }
    menu[split->group][split->member] = {static_cast<int>(i), split->flatName};
  }
  bool changed = false;
  for (auto& record : m_records) {
    auto group = menu.find(record.getGroupName());
    if (group == menu.end()) {
      // The whole group is gone from this run: every member of it fills as false.
      TriggerMenu gone;
      changed |= record.update(gone, writer, addProjections);
      continue;
    }
    changed |= record.update(group->second, writer, addProjections);
    menu.erase(group);
  }
  // A group with no record at all is a prefix that had no path in the menu the schema was built
  // from. Adding one would mean a new top-level field rather than growing a record, which this
  // module does not do.
  for (const auto& group : menu) {
    for (const auto& path : group.second) {
      edm::LogWarning("TriggerOutputFields") << "Skipping output of " << group.first << "." << path.first << ": no "
                                             << group.first << " path was in the menu the schema was built from.\n";
    }
  }
  return changed;
}

void TriggerOutputFields::fill(const edm::EventForOutput& event) {
  edm::Handle<edm::TriggerResults> handle;
  event.getByToken(m_token, handle);
  for (auto& record : m_records) {
    record.fill(*handle);
  }
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

std::string TriggerOutputFields::uniqueName(const RNTupleModel& model, const std::string& name) const {
  const auto& fieldNames = model.GetFieldNames();
  if (!fieldNames.contains(name)) {
    return name;
  }
  // The name is taken, usually by the same group from another process or by a table column. Append
  // the process, and keep going if even that is taken: the model must never be handed a name it
  // already holds, which RNTuple rejects outright, so the loop has to end on a free one.
  std::string unique = name + "_p" + m_processName;
  for (unsigned int i = 2; fieldNames.contains(unique); i++) {
    unique = name + "_p" + m_processName + "_" + std::to_string(i);
  }
  edm::LogWarning("TriggerOutputFields") << "Found a field named " << name << " already present; writing this one as "
                                         << unique << " instead.\n";
  return unique;
}
