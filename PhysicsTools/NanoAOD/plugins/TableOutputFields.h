#ifndef PhysicsTools_NanoAOD_TableOutputFields_h
#define PhysicsTools_NanoAOD_TableOutputFields_h

#include "RNTupleFieldPtr.h"

#include "FWCore/Framework/interface/EventForOutput.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <algorithm>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
using ROOT::Experimental::RCollectionNTupleWriter;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriter;

template <typename T>
class FlatTableField {
public:
  FlatTableField() = default;
  FlatTableField(const nanoaod::FlatTable& table, std::size_t i, RNTupleModel& model) {
    m_flatTableName = table.columnName(i);
    // case 1: field has a name (table may or may not have a name)
    if (!table.columnName(i).empty()) {
      m_field = RNTupleFieldPtr<T>(table.columnName(i), table.columnDoc(i), model);
      return;
    }
    // case 2: field doesn't have a name: use the table name as the RNTuple field name
    if (table.name().empty()) {
      throw cms::Exception("LogicError", "Empty FlatTable name and field name");
    }
    m_field = RNTupleFieldPtr<T>(table.name(), table.doc(), model);
  }
  // For collection fields and singleton fields
  void fill(const nanoaod::FlatTable& table, std::size_t i) {
    int col_idx = table.columnIndex(m_flatTableName);
    if (col_idx == -1) {
      throw cms::Exception("LogicError", "Missing column in input for " + table.name() + "_" + m_flatTableName);
    }
    m_field.fill(table.columnData<T>(col_idx)[i]);
  }
  // For vector fields without a collection, we have to buffer the results
  // internally and then fill the RNTuple field
  void fillVectored(const nanoaod::FlatTable& table) {
    int col_idx = table.columnIndex(m_flatTableName);
    if (col_idx == -1) {
      throw cms::Exception("LogicError", "Missing column in input for " + table.name() + "_" + m_flatTableName);
    }
    std::vector<typename T::value_type> buf(table.size());
    for (std::size_t i = 0; i < table.size(); i++) {
      buf[i] = table.columnData<typename T::value_type>(col_idx)[i];
    }
    m_field.fill(buf);
  }
  const std::string& getFlatTableName() const { return m_flatTableName; }

private:
  RNTupleFieldPtr<T> m_field;
  std::string m_flatTableName;
};

class TableOutputFields {
public:
  TableOutputFields() = default;
  explicit TableOutputFields(const edm::EDGetToken& token) : m_token(token) {}
  void print() const;
  void createFields(const edm::EventForOutput& event, RNTupleModel& model);
  void fillEntry(const nanoaod::FlatTable& table, std::size_t i);
  const edm::EDGetToken& getToken() const;

private:
  edm::EDGetToken m_token;
  std::vector<FlatTableField<float>> m_floatFields;
  std::vector<FlatTableField<int>> m_intFields;
  std::vector<FlatTableField<std::uint8_t>> m_uint8Fields;
  std::vector<FlatTableField<bool>> m_boolFields;
};

class TableOutputVectorFields {
public:
  TableOutputVectorFields() = default;
  explicit TableOutputVectorFields(const edm::EDGetToken& token) : m_token(token) {}
  void createFields(const edm::EventForOutput& event, RNTupleModel& model);
  void fill(const edm::EventForOutput& event);

private:
  edm::EDGetToken m_token;
  std::vector<FlatTableField<std::vector<float>>> m_vfloatFields;
  std::vector<FlatTableField<std::vector<int>>> m_vintFields;
  std::vector<FlatTableField<std::vector<std::uint8_t>>> m_vuint8Fields;
  std::vector<FlatTableField<std::vector<bool>>> m_vboolFields;
};

class TableCollection {
public:
  TableCollection() = default;
  // Invariants:
  // * table has a non-empty base name
  // * table has at least one column
  void add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table);
  // Invariants:
  // * m_main not null
  // * m_collectionName not empty
  void createFields(const edm::EventForOutput& event, RNTupleModel& eventModel);
  void fill(const edm::EventForOutput& event);
  void print() const;
  bool hasMainTable();
  const std::string& getCollectionName() const;

private:
  std::string m_collectionName;
  std::shared_ptr<RCollectionNTupleWriter> m_collection;
  TableOutputFields m_main;
  std::vector<TableOutputFields> m_extensions;
};

class TableCollectionSet {
public:
  void add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table);
  void createFields(const edm::EventForOutput& event, RNTupleModel& eventModel);
  void fill(const edm::EventForOutput& event);
  void print() const;

private:
  // Returns true if the FlatTable has an anonymous column. Throws a cms::Exception
  // if there is more than one anonymous column.
  static bool hasAnonymousColumn(const nanoaod::FlatTable& table);
  std::vector<TableCollection> m_collections;
  std::vector<TableOutputFields> m_singletonFields;
  std::vector<TableOutputVectorFields> m_vectorFields;
};

#endif
