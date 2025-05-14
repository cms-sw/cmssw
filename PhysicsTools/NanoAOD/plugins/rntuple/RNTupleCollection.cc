#include "RNTupleCollection.h"

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RField/RFieldRecord.hxx>
#include <ROOT/RField/RFieldSequenceContainer.hxx>

using ROOT::REntry;
using ROOT::RFieldBase;
using ROOT::RNTupleModel;
using ROOT::RRecordField;
using ROOT::RVectorField;

std::string flatTableColumnTypeToString(nanoaod::FlatTable::ColumnType type) {
  switch (type) {
    case nanoaod::FlatTable::ColumnType::UInt8:
      return "std::uint8_t";
    case nanoaod::FlatTable::ColumnType::Int16:
      return "std::int16_t";
    case nanoaod::FlatTable::ColumnType::UInt16:
      return "std::uint16_t";
    case nanoaod::FlatTable::ColumnType::Int32:
      return "std::int32_t";
    case nanoaod::FlatTable::ColumnType::UInt32:
      return "std::uint32_t";
    case nanoaod::FlatTable::ColumnType::Int64:
      return "std::int64_t";
    case nanoaod::FlatTable::ColumnType::UInt64:
      return "std::uint64_t";
    case nanoaod::FlatTable::ColumnType::Bool:
      return "bool";
    case nanoaod::FlatTable::ColumnType::Float:
      return "float";
    case nanoaod::FlatTable::ColumnType::Double:
      return "double";
    default:
      throw cms::Exception("LogicError", "Unsupported type");
  }
}

RNTupleCollection::RNTupleCollection(const std::string& name,
                                     const std::string& desc,
                                     std::vector<RNTupleSubfieldDescription>& subfields_desc,
                                     RNTupleModel& model)
    : m_name(name) {
  std::vector<std::unique_ptr<RFieldBase>> subfields;
  for (auto& sf_desc : subfields_desc) {
    std::string type = flatTableColumnTypeToString(sf_desc.m_type);
    // TODO: check how to add the description
    auto field = RFieldBase::Create(sf_desc.m_name, type).Unwrap();
    subfields.push_back(std::move(field));
  }
  auto record_field = std::make_unique<RRecordField>("_0", std::move(subfields));
  m_record_size = record_field->GetValueSize();
  m_record_offsets = record_field->GetOffsets();
  // TODO: check how to add the description
  auto collection_field = RVectorField::CreateUntyped(name, std::move(record_field));
}

void RNTupleCollection::bind_entry(REntry& entry) { entry.BindRawPtr<void>(m_name, &m_buffer); }
