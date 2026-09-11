#ifndef PhysicsTools_NanoAOD_RNTupleProjections_h
#define PhysicsTools_NanoAOD_RNTupleProjections_h

#include <ROOT/RNTupleModel.hxx>

#include <string>
#include <unordered_set>

// TTree-style flat names for the members of the grouped fields.
//
// This module writes a named collection as one untyped std::vector<record> and a trigger group or
// singleton table as one untyped record: GenJet.pt, HLT.IsoMu24, Generator.binvar. The TTree output
// module writes GenJet_pt, HLT_IsoMu24 and Generator_binvar instead. A projected field gives each
// member the TTree name too: a second name for the same columns, costing only a schema entry.
namespace rntupleprojection {
  // The flat name of a member: GenJet and pt give GenJet_pt. add() takes the name rather than
  // deriving it, because not every path's flat name is this join; see splitTriggerName().
  std::string projectedName(const std::string& fieldName, const std::string& memberName);

  // Project the member `memberName` of the grouped field `fieldName` as the top-level field `name`.
  // `inCollection` says whether the record sits inside a vector, in which case the projection is a
  // vector of the member type rather than a scalar. Returns whether the projection was added; a
  // name already in the model is skipped with a warning, because RNTuple throws on a duplicate
  // rather than reporting it.
  //
  // For a model that is not being written yet.
  bool add(ROOT::RNTupleModel& model,
           const std::string& name,
           const std::string& fieldName,
           const std::string& memberName,
           const std::string& memberTypeName,
           const std::string& memberDescription,
           bool inCollection);
  // For an ntuple that is already being written; the updater must have an open transaction. The
  // model cannot be asked for its names here, so `taken` carries them: seed it from
  // GetFieldNames() before the transaction is opened, and a name used is inserted into it.
  bool add(ROOT::RNTupleModel::RUpdater& updater,
           std::unordered_set<std::string>& taken,
           const std::string& name,
           const std::string& fieldName,
           const std::string& memberName,
           const std::string& memberTypeName,
           const std::string& memberDescription,
           bool inCollection);

  // Project every member of the grouped field `fieldName` under projectedName(), in the order the
  // record holds them. Does nothing if the model has no such field, or if it is neither an untyped
  // record nor a vector of one.
  void addForField(ROOT::RNTupleModel& model, const std::string& fieldName);
}  // namespace rntupleprojection

#endif
