#include "RNTupleCollection.h"

#include "FlatTableColumnDispatch.h"

#include <ROOT/RField.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RField/RFieldRecord.hxx>
#include <ROOT/RField/RFieldSequenceContainer.hxx>

using ROOT::REntry;
using ROOT::RFieldBase;
using ROOT::RNTupleModel;
using ROOT::RRecordField;
using ROOT::RVectorField;

namespace {
  // The RNTuple type name for a FlatTable column, e.g. "std::uint8_t" or "float".
  std::string columnTypeName(nanoaod::FlatTable::ColumnType type) {
    return dispatchColumnType(type, [](auto tag) { return ROOT::RField<typename decltype(tag)::type>::TypeName(); });
  }

  // Where a column's values start in the FlatTable, and how wide one value is.
  std::tuple<const unsigned char*, unsigned int> getColStartAndTypeSize(const nanoaod::FlatTable& table,
                                                                        unsigned int colIdx) {
    return dispatchColumnType(table.columnType(colIdx), [&](auto tag) {
      using ColumnT = typename decltype(tag)::type;
      auto column = table.columnData<ColumnT>(colIdx);
      return std::make_tuple(reinterpret_cast<const unsigned char*>(column.data()),
                             static_cast<unsigned int>(sizeof(nanoaod::FlatTable::ColumnStorageType<ColumnT>)));
    });
  }
}  // anonymous namespace

RNTupleCollection::RNTupleCollection(const std::string& name,
                                     const std::string& desc,
                                     std::vector<edm::Handle<nanoaod::FlatTable>>& tables,
                                     RNTupleModel& model)
    : m_name(name) {
  std::vector<std::unique_ptr<RFieldBase>> subfields;
  for (auto& table : tables) {
    for (unsigned int i = 0; i < table->nColumns(); i++) {
      auto field = RFieldBase::Create(table->columnName(i), columnTypeName(table->columnType(i))).Unwrap();
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
      auto [col_start, type_size] = getColStartAndTypeSize(*table, i);
      size_t col_offset = m_record_offsets[col_idx];

      for (unsigned int j = 0; j < col_size; j++) {
        std::memcpy(m_buffer.data() + (j * m_record_size) + col_offset, col_start + (j * type_size), type_size);
      }

      col_idx++;
    }
  }
}
