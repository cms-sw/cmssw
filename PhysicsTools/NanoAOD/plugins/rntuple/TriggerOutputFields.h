#ifndef PhysicsTools_NanoAOD_TriggerOutputFields_h
#define PhysicsTools_NanoAOD_TriggerOutputFields_h

#include "FWCore/Utilities/interface/EDGetToken.h"

#include <ROOT/REntry.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {
  class EventForOutput;
  class TriggerResults;
}  // namespace edm

// One path of the current menu: where it sits in the TriggerResults, and the flat name the TTree
// output module gives its branch, which is not always the record and member joined by "_"; see
// splitTriggerName().
struct TriggerMenuEntry {
  int index;
  std::string flatName;
};
// The paths of one group, keyed on the member name they were cut down to.
using TriggerMenu = std::map<std::string, TriggerMenuEntry>;

// One group of trigger-like paths -- HLT, Flag or L1 -- written as a single untyped record field
// whose members are the paths: HLT_IsoMu24 becomes HLT.IsoMu24, the same relation the collection
// fields have to the TTree module's Muon_pt branches.
//
// Grouping is what makes a path that only shows up in a later run recoverable: ROOT 6.40 can add a
// subfield to an untyped record of a bare model while the ntuple is being written, and entries
// already written read back false for it. Before 6.40 the record cannot grow, and such a path is
// written as a top-level bool field of its own instead, under the flat name it would have been
// projected to (HLT_IsoMu24); it is then in the file, but outside the record.
class TriggerRecordFields {
public:
  TriggerRecordFields(const std::string& groupName, const std::string& processName);

  const std::string& getGroupName() const { return m_groupName; }
  const std::string& getFieldName() const { return m_fieldName; }
  bool empty() const { return m_members.empty(); }

  // Collect a path before the record is built. Two paths that trim to one member name share it.
  void addPath(const std::string& member, const TriggerMenuEntry& path);
  // Build the record from the paths collected so far. The field name may differ from the group
  // name if something else in the model already claimed it; see TriggerOutputFields::uniqueName.
  void createField(ROOT::RNTupleModel& model, const std::string& fieldName);
  // Give each path the flat name the TTree module uses: HLT_IsoMu24 beside HLT.IsoMu24.
  void addProjections(ROOT::RNTupleModel& model) const;
  // Size the buffer from the record as the model now has it and point the entry at it. Must be
  // redone after every schema update: the record grows and the entries are invalidated.
  void bind(ROOT::REntry& entry, const ROOT::RNTupleModel& model);
  // Re-point the members at their index in this run's menu, -1 if the path is gone, and add fields
  // for paths that are new. Consumes the entries it claims. Returns true if the schema changed.
  bool update(TriggerMenu& menu, ROOT::RNTupleWriter& writer, bool addProjections);
  void fill(const edm::TriggerResults& triggers);

private:
  std::string m_groupName;
  std::string m_fieldName;
  std::string m_processName;
  // Member names in record order, their flat names, their index in the current TriggerResults, and
  // a lookup from the member name to its position in all three.
  std::vector<std::string> m_members;
  std::vector<std::string> m_flatNames;
  std::vector<int> m_indices;
  std::map<std::string, std::size_t> m_positions;
  // The record is filled through one flat buffer, as the collection fields are.
  std::vector<std::size_t> m_offsets;
  std::vector<unsigned char> m_buffer;
  // Paths the record could not take, on a ROOT too old to grow it: member name, name of the
  // top-level field standing in for the member, index in the current TriggerResults, and the value
  // that field is filled from.
  std::vector<std::string> m_looseMembers;
  std::vector<std::string> m_looseFlatNames;
  std::vector<int> m_looseIndices;
  std::vector<std::shared_ptr<bool>> m_looseValues;
};

class TriggerOutputFields {
public:
  TriggerOutputFields() = default;
  explicit TriggerOutputFields(const std::string& processName, const edm::EDGetToken& token)
      : m_token(token), m_lastRun(-1), m_processName(processName) {}
  void createFields(const edm::EventForOutput& event, ROOT::RNTupleModel& model);
  void addProjections(ROOT::RNTupleModel& model) const;
  void bind(ROOT::REntry& entry, const ROOT::RNTupleModel& model);
  // Called on every event before fill(): on a run boundary the menu is re-read, which can change
  // the schema. Returns true if it did, in which case the caller has to bind a fresh entry.
  bool updateForRun(const edm::EventForOutput& event, ROOT::RNTupleWriter& writer, bool addProjections);
  void fill(const edm::EventForOutput& event);

private:
  static std::vector<std::string> getTriggerNames(const edm::TriggerResults& triggerResults);
  // A record name not already taken in the model, warning if it had to be changed.
  std::string uniqueName(const ROOT::RNTupleModel& model, const std::string& name) const;

  edm::EDGetToken m_token;
  long m_lastRun;
  std::string m_processName;
  std::vector<TriggerRecordFields> m_records;
};

#endif
