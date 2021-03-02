#ifndef PhysicsTools_NanoAOD_TableOutputFields_h
#define PhysicsTools_NanoAOD_TableOutputFields_h

#include "RNTupleFieldPtr.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <algorithm>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriter;
using ROOT::Experimental::RCollectionNTuple;

void print_table(const nanoaod::FlatTable& table) {
  std::cout << "FlatTable {\n";
  std::cout << "  name: " << (table.name().empty() ? "// anon" : table.name()) << "\n";
  std::cout << "  singleton: " << (table.singleton() ? "true" : "false") << "\n";
  std::cout << "  size: " << table.size() << "\n";
  std::cout << "  extension: " << (table.extension() ? "true" : "false") << "\n";
  std::cout << "  fields: {\n";
  for (std::size_t i = 0; i < table.nColumns(); i++) {
    std::cout << "    " << (table.columnName(i).empty() ? "// anon" : table.columnName(i)) << ": ";
    switch(table.columnType(i)) {
      case nanoaod::FlatTable::ColumnType::Float:
        std::cout << "f32,"; break;
      case nanoaod::FlatTable::ColumnType::Int:
        std::cout << "i32,"; break;
      case nanoaod::FlatTable::ColumnType::UInt8:
        std::cout << "u8,"; break;
      case nanoaod::FlatTable::ColumnType::Bool:
        std::cout << "bool,"; break;
      default:
        throw cms::Exception("LogicError", "Unsupported type");
    }
    std::cout << "\n";
  }
  std::cout << "  }\n}\n";
}

class TableOutputFields {
public:
  TableOutputFields() = default;
  explicit TableOutputFields(const edm::EDGetToken& token) //, const nanoaod::FlatTable& table)
        : m_token(token)
  {

  }
  void print() const {
    for (const auto& field: m_floatFields) {
      std::cout << "  " << field.getFieldName() << ": f32,\n";
    }
    for (const auto& field: m_intFields) {
      std::cout << "  " << field.getFieldName() << ": i32,\n";
    }
    for (const auto& field: m_uint8Fields) {
      std::cout << "  " << field.getFieldName() << ": u8,\n";
    }
    for (const auto& field: m_boolFields) {
      std::cout << "  " << field.getFieldName() << ": bool,\n";
    }
  }
  void createFields(const edm::EventForOutput& event, RNTupleModel& model) {
    edm::Handle<nanoaod::FlatTable> handle;
    event.getByToken(m_token, handle);
    const nanoaod::FlatTable& table = *handle;
    for (std::size_t i = 0; i < table.nColumns(); i++) {
      switch(table.columnType(i)) {
        case nanoaod::FlatTable::ColumnType::Float:
          m_floatFields.emplace_back(RNTupleFieldPtr<float>(table.columnName(i), model)); break;
        case nanoaod::FlatTable::ColumnType::Int:
          m_intFields.emplace_back(RNTupleFieldPtr<int>(table.columnName(i), model)); break;
        case nanoaod::FlatTable::ColumnType::UInt8:
          m_uint8Fields.emplace_back(RNTupleFieldPtr<std::uint8_t>(table.columnName(i), model)); break;
        case nanoaod::FlatTable::ColumnType::Bool:
          m_boolFields.emplace_back(RNTupleFieldPtr<bool>(table.columnName(i), model)); break;
        default:
          throw cms::Exception("LogicError", "Unsupported type");
      }
    }
  }
  void fillEntry(const nanoaod::FlatTable& table, std::size_t i) {
    auto column_index = [&](const std::string& field_name) -> int {
      int col_idx = table.columnIndex(field_name);
      if (col_idx == -1) {
        // todo rntuple naming?
        throw cms::Exception("LogicError", "Missing column in input for "
          + table.name() + "_" + field_name);
      }
      return col_idx;
    };
    for (auto& field: m_floatFields) {
      field.fill(table.columnData<float>(column_index(field.getFieldName()))[i]);
    }
    for (auto& field: m_intFields) {
      field.fill(table.columnData<int>(column_index(field.getFieldName()))[i]);
    }
    for (auto& field: m_uint8Fields) {
      field.fill(table.columnData<std::uint8_t>(column_index(field.getFieldName()))[i]);
    }
    for (auto& field: m_boolFields) {
      field.fill(table.columnData<bool>(column_index(field.getFieldName()))[i]);
    }
  }
  const edm::EDGetToken& getToken() const {
    return m_token;
  }

private:
  edm::EDGetToken m_token;
  std::vector<RNTupleFieldPtr<float>> m_floatFields;
  std::vector<RNTupleFieldPtr<int>> m_intFields;
  std::vector<RNTupleFieldPtr<std::uint8_t>> m_uint8Fields;
  std::vector<RNTupleFieldPtr<bool>> m_boolFields;
};

class TableCollection {
  public:
    TableCollection() = default;
    // invariants:
    // * table has a non-empty base name
    // * table has at least one column
    void add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table) {
      if (m_collectionName.empty()) {
        m_collectionName = table.name();
        // TODO ensure this isn't moved from twice?
      }
      if (table.extension()) {
        m_extensions.emplace_back(TableOutputFields(table_token));
        return;
      }
      if (hasMainTable()) {
        throw cms::Exception("LogicError", "Trying to save multiple main tables for " + m_collectionName + "\n");
     }
      m_main = TableOutputFields(table_token);
    }
    // Invariants:
    // * m_main not null
    // * m_collectionName not empty
    void createFields(const edm::EventForOutput& event, RNTupleModel& eventModel) {
      auto collectionModel = RNTupleModel::Create();
      m_main.createFields(event, *collectionModel);
      for (auto& extension: m_extensions) {
        extension.createFields(event, *collectionModel);
      }
      m_collection = eventModel.MakeCollection(m_collectionName, std::move(collectionModel));
    }
    void fill(const edm::EventForOutput& event) {
      edm::Handle<nanoaod::FlatTable> handle;
      event.getByToken(m_main.getToken(), handle);
      const auto& main_table = *handle;
      auto table_size = main_table.size();
      // todo check table sizes
      for (std::size_t i = 0; i < table_size; i++) {
        m_main.fillEntry(main_table, i);
        for (auto& ext: m_extensions) {
          edm::Handle<nanoaod::FlatTable> handle;
          event.getByToken(ext.getToken(), handle);
          const auto& ext_table = *handle;
          ext.fillEntry(ext_table, i);
        }
        m_collection->Fill();
      }
      //if (num_fill != num_fill_ext) {
      //  throw cms::Exception("LogicError",
      //    "Mismatch in number of entries between extension and main table for " + m_collectionName);
      //}
    }
    void print() const {
      std::cout << "Collection: " << m_collectionName << " {\n";
      m_main.print();
      for (const auto& ext: m_extensions) {
        ext.print();
      }
      std::cout << "}\n";
    }
    bool hasMainTable() {
      return !m_main.getToken().isUninitialized();
    }
    const std::string& getCollectionName() const {
      return m_collectionName;
    }
  private:
    std::string m_collectionName;
    std::shared_ptr<RCollectionNTuple> m_collection;
    TableOutputFields m_main;
    std::vector<TableOutputFields> m_extensions;
};

class TableCollections {
  public:
    void add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table) {
      // skip empty tables -- requirement of RNTuple to define schema before filling
      if (table.nColumns() == 0) {
        std::cout << "Warning: skipping empty table: \n";
        print_table(table);
        return;
      }
      // skip anonymous tables for now
      // TODO add to m_topLevelFields;
      if (table.name().empty()) {
        std::cout << "skipping anonymous table:\n";
        print_table(table);
        return;
      }
      // skip tables with anonymous fields for now
      // TODO add to m_topLevelFields;
      for (std::size_t i = 0; i < table.nColumns(); i++) {
        auto col_name = table.columnName(i).empty() ? "// anon" : table.columnName(i);
        if (table.columnName(i).empty()) {
          std::cout << "skipping table with anonymous fields:\n";
          print_table(table);
          return;
        }
      }
      auto collection = std::find_if(m_collections.begin(), m_collections.end(),
        [&](const TableCollection& c) { return c.getCollectionName() == table.name(); });
      if (collection == m_collections.end()) {
        std::cout << "adding new collection: \n";
        print_table(table);
        m_collections.emplace_back(TableCollection());
        m_collections.back().add(table_token, table);
        return;
      }
      std::cout << "adding to existing collection : \n";
      print_table(table);
      collection->add(table_token, table);
    }
    void print() const {
      for (const auto& collection: m_collections) {
        collection.print();
        std::cout << "\n";
      }
    }
    void createFields(const edm::EventForOutput& event, RNTupleModel& eventModel) {
      for (auto& collection: m_collections) {
        if (!collection.hasMainTable()) {
          throw cms::Exception("LogicError", "Trying to save an extension table for " +
            collection.getCollectionName() + " without the corresponding main table\n");
        }
        collection.createFields(event, eventModel);
      }
      // todo top-level fields
    }
    void fill(const edm::EventForOutput& event) {
      for (auto& collection: m_collections) {
        collection.fill(event);
      }
      // todo top-level fields
    }
  private:
    std::vector<TableCollection> m_collections;
    std::vector<TableOutputFields> m_topLevelFields;
};

#endif
