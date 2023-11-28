#include "TableOutputFields.h"

namespace {
  void printTable(const nanoaod::FlatTable& table) {
    std::cout << "FlatTable {\n";
    std::cout << "  name: " << (table.name().empty() ? "// anon" : table.name()) << "\n";
    std::cout << "  singleton: " << (table.singleton() ? "true" : "false") << "\n";
    std::cout << "  size: " << table.size() << "\n";
    std::cout << "  extension: " << (table.extension() ? "true" : "false") << "\n";
    std::cout << "  fields: {\n";
    for (std::size_t i = 0; i < table.nColumns(); i++) {
      std::cout << "    " << (table.columnName(i).empty() ? "// anon" : table.columnName(i)) << ": ";
      switch (table.columnType(i)) {
        case nanoaod::FlatTable::ColumnType::Float:
          std::cout << "f32,";
          break;
        case nanoaod::FlatTable::ColumnType::Int32:
          std::cout << "i32,";
          break;
        case nanoaod::FlatTable::ColumnType::UInt8:
          std::cout << "u8,";
          break;
        case nanoaod::FlatTable::ColumnType::Bool:
          std::cout << "bool,";
          break;
        default:
          throw cms::Exception("LogicError", "Unsupported type");
      }
      std::cout << "\n";
    }
    std::cout << "  }\n}\n";
  }
}  // anonymous namespace

void TableOutputFields::print() const {
  for (const auto& field : m_floatFields) {
    std::cout << "  " << field.getFlatTableName() << ": f32,\n";
  }
  for (const auto& field : m_intFields) {
    std::cout << "  " << field.getFlatTableName() << ": i32,\n";
  }
  for (const auto& field : m_uint8Fields) {
    std::cout << "  " << field.getFlatTableName() << ": u8,\n";
  }
  for (const auto& field : m_boolFields) {
    std::cout << "  " << field.getFlatTableName() << ": bool,\n";
  }
}

void TableOutputFields::createFields(const edm::EventForOutput& event, RNTupleModel& model) {
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_token, handle);
  const nanoaod::FlatTable& table = *handle;
  for (std::size_t i = 0; i < table.nColumns(); i++) {
    switch (table.columnType(i)) {
      case nanoaod::FlatTable::ColumnType::Float:
        m_floatFields.emplace_back(FlatTableField<float>(table, i, model));
        break;
      case nanoaod::FlatTable::ColumnType::Int32:
        m_intFields.emplace_back(FlatTableField<int>(table, i, model));
        break;
      case nanoaod::FlatTable::ColumnType::UInt8:
        m_uint8Fields.emplace_back(FlatTableField<std::uint8_t>(table, i, model));
        break;
      case nanoaod::FlatTable::ColumnType::Bool:
        m_boolFields.emplace_back(FlatTableField<bool>(table, i, model));
        break;
      default:
        throw cms::Exception("LogicError", "Unsupported type");
    }
  }
}

void TableOutputFields::fillEntry(const nanoaod::FlatTable& table, std::size_t i) {
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

const edm::EDGetToken& TableOutputFields::getToken() const { return m_token; }

///////////////////////////////////////////////////////////////////////////////

void TableOutputVectorFields::createFields(const edm::EventForOutput& event, RNTupleModel& model) {
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_token, handle);
  const nanoaod::FlatTable& table = *handle;
  for (std::size_t i = 0; i < table.nColumns(); i++) {
    switch (table.columnType(i)) {
      case nanoaod::FlatTable::ColumnType::Float:
        m_vfloatFields.emplace_back(FlatTableField<std::vector<float>>(table, i, model));
        break;
      case nanoaod::FlatTable::ColumnType::Int32:
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
void TableOutputVectorFields::fill(const edm::EventForOutput& event) {
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

///////////////////////////////////////////////////////////////////////////////

void TableCollection::add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table) {
  if (m_collectionName.empty()) {
    m_collectionName = table.name();
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

void TableCollection::createFields(const edm::EventForOutput& event, RNTupleModel& eventModel) {
  auto collectionModel = RNTupleModel::Create();
  m_main.createFields(event, *collectionModel);
  for (auto& extension : m_extensions) {
    extension.createFields(event, *collectionModel);
  }
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_main.getToken(), handle);
  const nanoaod::FlatTable& table = *handle;
  collectionModel->SetDescription(table.doc());
  m_collection = eventModel.MakeCollection(m_collectionName, std::move(collectionModel));
}

void TableCollection::fill(const edm::EventForOutput& event) {
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_main.getToken(), handle);
  const auto& main_table = *handle;
  auto table_size = main_table.size();
  for (std::size_t i = 0; i < table_size; i++) {
    m_main.fillEntry(main_table, i);
    for (auto& ext : m_extensions) {
      edm::Handle<nanoaod::FlatTable> handle;
      event.getByToken(ext.getToken(), handle);
      const auto& ext_table = *handle;
      if (ext_table.size() != table_size) {
        throw cms::Exception("LogicError",
                             "Mismatch in number of entries between extension and main table for " + m_collectionName);
      }
      ext.fillEntry(ext_table, i);
    }
    m_collection->Fill();
  }
}

void TableCollection::print() const {
  std::cout << "Collection: " << m_collectionName << " {\n";
  m_main.print();
  for (const auto& ext : m_extensions) {
    ext.print();
  }
  std::cout << "}\n";
}

bool TableCollection::hasMainTable() { return !m_main.getToken().isUninitialized(); }

const std::string& TableCollection::getCollectionName() const { return m_collectionName; }

///////////////////////////////////////////////////////////////////////////////

void TableCollectionSet::add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table) {
  // skip empty tables -- requirement of RNTuple to define schema before filling
  if (table.nColumns() == 0) {
    std::cout << "Warning: skipping empty table: \n";
    printTable(table);
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
  auto collection = std::find_if(m_collections.begin(), m_collections.end(), [&](const TableCollection& c) {
    return c.getCollectionName() == table.name();
  });
  if (collection == m_collections.end()) {
    m_collections.emplace_back(TableCollection());
    m_collections.back().add(table_token, table);
    return;
  }
  collection->add(table_token, table);
}

void TableCollectionSet::print() const {
  for (const auto& collection : m_collections) {
    collection.print();
    std::cout << "\n";
  }
}

void TableCollectionSet::createFields(const edm::EventForOutput& event, RNTupleModel& eventModel) {
  for (auto& collection : m_collections) {
    if (!collection.hasMainTable()) {
      throw cms::Exception("LogicError",
                           "Trying to save an extension table for " + collection.getCollectionName() +
                               " without the corresponding main table\n");
    }
    collection.createFields(event, eventModel);
  }
  for (auto& table : m_singletonFields) {
    table.createFields(event, eventModel);
  }
  for (auto& table : m_vectorFields) {
    table.createFields(event, eventModel);
  }
}

void TableCollectionSet::fill(const edm::EventForOutput& event) {
  for (auto& collection : m_collections) {
    collection.fill(event);
  }
  for (auto& fields : m_singletonFields) {
    edm::Handle<nanoaod::FlatTable> handle;
    event.getByToken(fields.getToken(), handle);
    const auto& table = *handle;
    fields.fillEntry(table, 0);
  }
  for (auto& fields : m_vectorFields) {
    fields.fill(event);
  }
}

bool TableCollectionSet::hasAnonymousColumn(const nanoaod::FlatTable& table) {
  int num_anon = 0;
  for (std::size_t i = 0; i < table.nColumns(); i++) {
    if (table.columnName(i).empty()) {
      num_anon++;
    }
  }
  if (num_anon > 1) {
    throw cms::Exception("LogicError",
                         "FlatTable `" + table.name() + "` has " + std::to_string(num_anon) + "anonymous fields");
  }
  return num_anon;
}
