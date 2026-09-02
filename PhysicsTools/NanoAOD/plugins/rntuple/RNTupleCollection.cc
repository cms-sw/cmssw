#include "RNTupleCollection.h"

#include "FlatTableColumnDispatch.h"
#include "RNTupleProjections.h"

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
                                     RNTupleModel& model,
                                     bool singleton)
    : m_name(name), m_singleton(singleton) {
  std::vector<std::unique_ptr<RFieldBase>> subfields;
  for (auto& table : tables) {
    for (unsigned int i = 0; i < table->nColumns(); i++) {
      auto field = RFieldBase::Create(table->columnName(i), columnTypeName(table->columnType(i))).Unwrap();
      field->SetDescription(table->columnDoc(i));
      subfields.push_back(std::move(field));
    }
  }
  // A singleton's record is the top-level field itself and takes the collection name; the record
  // inside a vector is an item field, which RNTuple names "_0" by convention.
  auto record_field = std::make_unique<RRecordField>(m_singleton ? name : "_0", std::move(subfields));
  m_record_size = record_field->GetValueSize();
  m_record_offsets = record_field->GetOffsets();
  if (m_singleton) {
    record_field->SetDescription(desc);
    model.AddField(std::move(record_field));
    return;
  }
  auto collection_field = RVectorField::CreateUntyped(name, std::move(record_field));
  collection_field->SetDescription(desc);
  model.AddField(std::move(collection_field));
}

void RNTupleCollection::addProjections(RNTupleModel& model) const { rntupleprojection::addForField(model, m_name); }

void RNTupleCollection::bindBuffer(REntry& entry) {
  if (m_singleton) {
    // A record's value is the memory holding it, so the entry gets the buffer's contents; a vector
    // field's value is the std::vector itself, so it gets the buffer object. Sizing the record
    // buffer here, once, is what keeps the bound address valid: fill() must never reallocate it.
    m_buffer.resize(m_record_size);
    entry.BindRawPtr<void>(m_name, m_buffer.data());
    return;
  }
  entry.BindRawPtr<void>(m_name, &m_buffer);
}

void RNTupleCollection::fill(std::vector<edm::Handle<nanoaod::FlatTable>>& tables) {
  unsigned int col_idx = 0;
  size_t col_size = tables.empty() ? 0 : tables[0]->size();

  if (m_singleton) {
    // One row, always: the field has no place to put a second and no way to say there was none.
    if (col_size != 1) {
      throw cms::Exception("LogicError",
                           "Singleton table " + m_name + " has " + std::to_string(col_size) + " rows, expected 1");
    }
  } else {
    m_buffer.resize(m_record_size * col_size);
  }

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
