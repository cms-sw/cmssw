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

template<typename T>
class FlatTableField {
public:
  FlatTableField() = default;
  FlatTableField(const nanoaod::FlatTable& table, std::size_t i, RNTupleModel& model) {
    m_flatTableName = table.columnName(i);
    // case 1: field has a name (table may or may not have a name)
    if (!table.columnName(i).empty()) {
      m_field = RNTupleFieldPtr<T>(table.columnName(i), model);
      return;
    }
    // case 2: field doesn't have a name: use the table name as the RNTuple field name
    if (table.name().empty()) {
      throw cms::Exception("LogicError", "Empty FlatTable name and field name");
    }
    m_field = RNTupleFieldPtr<T>(table.name(), model);
  }
  // For collection fields and singleton fields
  void fill(const nanoaod::FlatTable& table, std::size_t i) {
    int col_idx = table.columnIndex(m_flatTableName);
    if (col_idx == -1) {
      throw cms::Exception("LogicError", "Missing column in input for "
        + table.name() + "_" + m_flatTableName);
    }
    m_field.fill(table.columnData<T>(col_idx)[i]);
  }
  // For vector fields without a collection, we have to buffer the results
  // internally and then fill the RNTuple field
  void fillVectored(const nanoaod::FlatTable& table) {
    int col_idx = table.columnIndex(m_flatTableName);
    if (col_idx == -1) {
      throw cms::Exception("LogicError", "Missing column in input for "
        + table.name() + "_" + m_flatTableName);
    }
    std::vector<typename T::value_type> buf(table.size());
    for (std::size_t i = 0; i < table.size(); i++) {
      buf[i] = table.columnData<typename T::value_type>(col_idx)[i];
    }
    m_field.fill(buf);
  }
  const std::string& getFlatTableName() const {
    return m_flatTableName;
  }
private:
  RNTupleFieldPtr<T> m_field;
  std::string m_flatTableName;
};

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
        : m_token(token) {}
  void print() const {
    for (const auto& field: m_floatFields) {
      std::cout << "  " << field.getFlatTableName() << ": f32,\n";
    }
    for (const auto& field: m_intFields) {
      std::cout << "  " << field.getFlatTableName() << ": i32,\n";
    }
    for (const auto& field: m_uint8Fields) {
      std::cout << "  " << field.getFlatTableName() << ": u8,\n";
    }
    for (const auto& field: m_boolFields) {
      std::cout << "  " << field.getFlatTableName() << ": bool,\n";
    }
  }
  void createFields(const edm::EventForOutput& event, RNTupleModel& model) {
    edm::Handle<nanoaod::FlatTable> handle;
    event.getByToken(m_token, handle);
    const nanoaod::FlatTable& table = *handle;
    for (std::size_t i = 0; i < table.nColumns(); i++) {
      switch(table.columnType(i)) {
        case nanoaod::FlatTable::ColumnType::Float:
          m_floatFields.emplace_back(FlatTableField<float>(table, i, model)); break;
        case nanoaod::FlatTable::ColumnType::Int:
          m_intFields.emplace_back(FlatTableField<int>(table, i, model)); break;
        case nanoaod::FlatTable::ColumnType::UInt8:
          m_uint8Fields.emplace_back(FlatTableField<std::uint8_t>(table, i, model)); break;
        case nanoaod::FlatTable::ColumnType::Bool:
          m_boolFields.emplace_back(FlatTableField<bool>(table, i, model)); break;
        default:
          throw cms::Exception("LogicError", "Unsupported type");
      }
    }
  }
  void fillEntry(const nanoaod::FlatTable& table, std::size_t i) {
    for (auto& field : m_floatFields) {
      field.fill(table, i);
    }
    for (auto& field : m_intFields) {
      field.fill(table, i);
    }
    for (auto& field : m_uint8Fields) {
      field.fill(table, i);
    }
    for (auto& field : m_boolFields) {
      field.fill(table, i);
    }
  }
  const edm::EDGetToken& getToken() const {
    return m_token;
  }

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
  explicit TableOutputVectorFields(const edm::EDGetToken& token)
        : m_token(token) {}
  void createFields(const edm::EventForOutput& event, RNTupleModel& model) {
    edm::Handle<nanoaod::FlatTable> handle;
    event.getByToken(m_token, handle);
    const nanoaod::FlatTable& table = *handle;
    for (std::size_t i = 0; i < table.nColumns(); i++) {
      switch(table.columnType(i)) {
        case nanoaod::FlatTable::ColumnType::Float:
          m_vfloatFields.emplace_back(FlatTableField<std::vector<float>>(table, i, model));
          break;
        case nanoaod::FlatTable::ColumnType::Int:
          m_vintFields.emplace_back(FlatTableField<std::vector<int>>(table, i, model));
          break;
        case nanoaod::FlatTable::ColumnType::UInt8:
          m_vuint8Fields.emplace_back(FlatTableField<std::vector<std::uint8_t>>(table, i, model));
          break;
        case nanoaod::FlatTable::ColumnType::Bool:
          m_vboolFields.emplace_back(FlatTableField<std::vector<bool>>(table, i, model));
          break;
        default:
          throw cms::Exception("LogicError", "Unsupported type");
      }
    }
  }
  void fill(const edm::EventForOutput& event) {
    edm::Handle<nanoaod::FlatTable> handle;
    event.getByToken(m_token, handle);
    const auto& table = *handle;
    for (auto& field : m_vfloatFields) {
      field.fillVectored(table);
    }
    for (auto& field : m_vintFields) {
      field.fillVectored(table);
    }
    for (auto& field : m_vuint8Fields) {
      field.fillVectored(table);
    }
    for (auto& field : m_vboolFields) {
      field.fillVectored(table);
    }
  }
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
      // Can handle either anonymous table or anonymous column but not both
      // - anonymous table: use column names directly as top-level fields
      // - anonymous column: use the table name as the field name
      if (table.name().empty() && hasAnonymousColumn(table)) {
        throw cms::Exception("LogicError", "Anonymous FlatTable and anonymous field");
      }
      // case 1: create a top-level RNTuple field for each table column
      if (table.name().empty() || hasAnonymousColumn(table)) {
        if (table.singleton()) {
          m_singletonFields.emplace_back(TableOutputFields(table_token));
        } else {
          m_vectorFields.emplace_back(TableOutputVectorFields(table_token));
        }
        return;
      }
      // case 2: Named singleton and vector tables are both written as RNTuple collections.
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
      for (auto& table: m_singletonFields) {
        table.createFields(event, eventModel);
      }
      for (auto& table: m_vectorFields) {
        table.createFields(event, eventModel);
      }
    }
    void fill(const edm::EventForOutput& event) {
      for (auto& collection: m_collections) {
        collection.fill(event);
      }
      for (auto& fields: m_singletonFields) {
        edm::Handle<nanoaod::FlatTable> handle;
        event.getByToken(fields.getToken(), handle);
        const auto& table = *handle;
        fields.fillEntry(table, 0);
      }
      for (auto& fields: m_vectorFields) {
        fields.fill(event);
      }
    }
  private:
    // Returns true if the FlatTable has an anonymous column. Throws a cms::Exception
    // if there is more than one anonymous column.
    static bool hasAnonymousColumn(const nanoaod::FlatTable& table) {
      int num_anon = 0;
      for (std::size_t i = 0; i < table.nColumns(); i++) {
        if (table.columnName(i).empty()) {
          num_anon++;
        }
      }
      if (num_anon > 1) {
        throw cms::Exception("LogicError", "FlatTable `" + table.name() +
          "` has " + std::to_string(num_anon) + "anonymous fields");
      }
      return num_anon;
    }
    std::vector<TableCollection> m_collections;
    std::vector<TableOutputFields> m_singletonFields;
    std::vector<TableOutputVectorFields> m_vectorFields;
};

#endif
