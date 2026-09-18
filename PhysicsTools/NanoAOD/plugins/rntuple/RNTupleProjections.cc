#include "RNTupleProjections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <ROOT/RField/RFieldRecord.hxx>
#include <ROOT/RFieldBase.hxx>

#include <memory>
#include <utility>
#include <vector>

using ROOT::RFieldBase;
using ROOT::RNTupleModel;
using ROOT::RRecordField;

namespace {

  // The projection field and the mapping from its names onto the source names.
  //
  // A member of a plain record maps straight onto it: HLT_IsoMu24 is HLT.IsoMu24. A member of a
  // record inside a vector needs both levels mapped, because the projection is a vector too: the
  // vector itself onto the collection, and its item onto the member. RNTuple names the item field
  // "_0" on either side, so the mapping is only ever asked about those two names.
  std::pair<std::unique_ptr<RFieldBase>, RNTupleModel::FieldMappingFunc_t> makeProjection(
      const std::string& name,
      const std::string& fieldName,
      const std::string& memberName,
      const std::string& memberTypeName,
      const std::string& memberDescription,
      bool inCollection) {
    auto field =
        RFieldBase::Create(name, inCollection ? "std::vector<" + memberTypeName + ">" : memberTypeName).Unwrap();
    field->SetDescription(memberDescription);
    const std::string member = fieldName + (inCollection ? "._0." : ".") + memberName;
    RNTupleModel::FieldMappingFunc_t mapping =
        [name, fieldName, member, inCollection](const std::string& projected) -> std::string {
      return (inCollection && projected == name) ? fieldName : member;
    };
    return {std::move(field), std::move(mapping)};
  }

  void warnDuplicate(const std::string& name, const std::string& fieldName, const std::string& memberName) {
    edm::LogWarning("RNTupleProjections") << "Not projecting " << fieldName << "." << memberName << " to " << name
                                          << ": a field of that name is already there.\n";
  }

  void warnRefused(const std::string& name,
                   const std::string& fieldName,
                   const std::string& memberName,
                   const std::string& report) {
    edm::LogWarning("RNTupleProjections")
        << "Could not project " << fieldName << "." << memberName << " to " << name << ": " << report << "\n";
  }

}  // anonymous namespace

std::string rntupleprojection::projectedName(const std::string& fieldName, const std::string& memberName) {
  return fieldName + "_" + memberName;
}

bool rntupleprojection::add(RNTupleModel& model,
                            const std::string& name,
                            const std::string& fieldName,
                            const std::string& memberName,
                            const std::string& memberTypeName,
                            const std::string& memberDescription,
                            bool inCollection) {
  if (model.GetFieldNames().contains(name)) {
    warnDuplicate(name, fieldName, memberName);
    return false;
  }
  auto [projection, mapping] =
      makeProjection(name, fieldName, memberName, memberTypeName, memberDescription, inCollection);
  auto result = model.AddProjectedField(std::move(projection), std::move(mapping));
  if (!result) {
    warnRefused(name, fieldName, memberName, result.GetError()->GetReport());
    return false;
  }
  return true;
}

bool rntupleprojection::add(RNTupleModel::RUpdater& updater,
                            std::unordered_set<std::string>& taken,
                            const std::string& name,
                            const std::string& fieldName,
                            const std::string& memberName,
                            const std::string& memberTypeName,
                            const std::string& memberDescription,
                            bool inCollection) {
  if (!taken.insert(name).second) {
    warnDuplicate(name, fieldName, memberName);
    return false;
  }
  auto [projection, mapping] =
      makeProjection(name, fieldName, memberName, memberTypeName, memberDescription, inCollection);
  auto result = updater.AddProjectedField(std::move(projection), std::move(mapping));
  if (!result) {
    warnRefused(name, fieldName, memberName, result.GetError()->GetReport());
    return false;
  }
  return true;
}

void rntupleprojection::addForField(RNTupleModel& model, const std::string& fieldName) {
  if (!model.GetFieldNames().contains(fieldName)) {
    return;
  }
  // A grouped field is the untyped record itself, or a vector whose one item field is one.
  const auto& field = model.GetConstField(fieldName);
  const auto* record = dynamic_cast<const RRecordField*>(&field);
  bool inCollection = false;
  if (nullptr == record) {
    const auto subfields = field.GetConstSubfields();
    if (subfields.size() != 1) {
      return;
    }
    record = dynamic_cast<const RRecordField*>(subfields[0]);
    inCollection = true;
  }
  if (nullptr == record) {
    return;
  }

  // What each member needs, collected before anything is added: adding a field to the model can
  // rehash the name set this walk reads.
  struct Member {
    std::string name;
    std::string typeName;
    std::string description;
  };
  const auto recordMembers = record->GetConstSubfields();
  std::vector<Member> members;
  members.reserve(recordMembers.size());
  for (const auto* member : recordMembers) {
    // Only a leaf gets a flat name; nothing this module writes nests a record inside a record.
    if (member->GetTypeName().empty()) {
      continue;
    }
    members.push_back({member->GetFieldName(), member->GetTypeName(), member->GetDescription()});
  }

  for (const auto& member : members) {
    add(model,
        projectedName(fieldName, member.name),
        fieldName,
        member.name,
        member.typeName,
        member.description,
        inCollection);
  }
}
