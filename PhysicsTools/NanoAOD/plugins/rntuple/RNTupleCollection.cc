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

std::tuple<const unsigned char*, unsigned int> getColStartAndTypeSize(edm::Handle<nanoaod::FlatTable>& table,
                                                                      unsigned int colIdx) {
  const unsigned char* col_start;
  switch (table->columnType(colIdx)) {
    case nanoaod::FlatTable::ColumnType::UInt8:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<uint8_t>(colIdx).data());
      return std::make_tuple(col_start, 1);
    case nanoaod::FlatTable::ColumnType::Int16:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<int16_t>(colIdx).data());
      return std::make_tuple(col_start, 2);
    case nanoaod::FlatTable::ColumnType::UInt16:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<uint16_t>(colIdx).data());
      return std::make_tuple(col_start, 2);
    case nanoaod::FlatTable::ColumnType::Int32:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<int32_t>(colIdx).data());
      return std::make_tuple(col_start, 4);
    case nanoaod::FlatTable::ColumnType::UInt32:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<uint32_t>(colIdx).data());
      return std::make_tuple(col_start, 4);
    case nanoaod::FlatTable::ColumnType::Int64:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<int64_t>(colIdx).data());
      return std::make_tuple(col_start, 8);
    case nanoaod::FlatTable::ColumnType::UInt64:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<uint64_t>(colIdx).data());
      return std::make_tuple(col_start, 8);
    case nanoaod::FlatTable::ColumnType::Bool:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<bool>(colIdx).data());
      return std::make_tuple(col_start, 1);
    case nanoaod::FlatTable::ColumnType::Float:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<float>(colIdx).data());
      return std::make_tuple(col_start, 4);
    case nanoaod::FlatTable::ColumnType::Double:
      col_start = reinterpret_cast<const unsigned char*>(table->columnData<double>(colIdx).data());
      return std::make_tuple(col_start, 8);
    default:
      throw cms::Exception("LogicError", "Unsupported type");
  }
}

RNTupleCollection::RNTupleCollection(const std::string& name,
                                     const std::string& desc,
                                     std::vector<edm::Handle<nanoaod::FlatTable>>& tables,
                                     RNTupleModel& model)
    : m_name(name) {
  std::vector<std::unique_ptr<RFieldBase>> subfields;
  for (auto& table : tables) {
    for (unsigned int i = 0; i < table->nColumns(); i++) {
      std::string type = flatTableColumnTypeToString(table->columnType(i));
      auto field = RFieldBase::Create(table->columnName(i), type).Unwrap();
      field->SetDescription(table->columnDoc(i));
      subfields.push_back(std::move(field));
    }
  }
  auto record_field = std::make_unique<RRecordField>("_0", std::move(subfields));
  m_record_size = record_field->GetValueSize();
  m_record_offsets = record_field->GetOffsets();
  auto collection_field = RVectorField::CreateUntyped(name, std::move(record_field));
  collection_field->SetDescription(desc);
  model.AddField(std::move(collection_field));
}

void RNTupleCollection::bindBuffer(RNTupleModel& model) {
  auto& default_entry = model.GetDefaultEntry();
  default_entry.BindRawPtr<void>(m_name, &m_buffer);
}

void RNTupleCollection::fill(std::vector<edm::Handle<nanoaod::FlatTable>>& tables) {
  unsigned int col_idx = 0;
  size_t col_size = tables.empty() ? 0 : tables[0]->size();

  m_buffer.resize(m_record_size * col_size);

  for (auto& table : tables) {
    if (table->size() != col_size) {
      throw cms::Exception("LogicError",
                           "Mismatch in number of entries between extension and main table for " + m_name);
    }
    for (unsigned int i = 0; i < table->nColumns(); i++) {
      auto [col_start, type_size] = getColStartAndTypeSize(table, i);
      size_t col_offset = m_record_offsets[col_idx];

      for (unsigned int j = 0; j < col_size; j++) {
        std::memcpy(m_buffer.data() + (j * m_record_size) + col_offset, col_start + (j * type_size), type_size);
      }

      col_idx++;
    }
  }
}
