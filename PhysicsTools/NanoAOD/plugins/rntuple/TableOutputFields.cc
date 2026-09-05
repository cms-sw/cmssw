#include "TableOutputFields.h"

#include "FlatTableColumnDispatch.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

using ROOT::RNTupleModel;

namespace flattablefield {
  std::pair<std::string, std::string> nameAndDoc(const nanoaod::FlatTable& table, std::size_t i) {
    // case 1: the column has a name (the table may or may not have one)
    if (!table.columnName(i).empty()) {
      return {table.columnName(i), table.columnDoc(i)};
    }
    // case 2: the column is anonymous, so it takes the table's name
    if (table.name().empty()) {
      throw cms::Exception("LogicError", "Empty FlatTable name and field name");
    }
    return {table.name(), table.doc()};
  }

  unsigned int columnIndex(const nanoaod::FlatTable& table, const std::string& columnName) {
    int index = table.columnIndex(columnName);
    if (index == -1) {
      throw cms::Exception("LogicError", "Missing column in input for " + table.name() + "_" + columnName);
    }
    return static_cast<unsigned int>(index);
  }
}  // namespace flattablefield

void TableOutputFields::createFields(const edm::OccurrenceForOutput& event, RNTupleModel& model) {
  const nanoaod::FlatTable& table = *getTable(event);
  m_fields.reserve(table.nColumns());
  for (std::size_t i = 0; i < table.nColumns(); i++) {
    dispatchColumnType(table.columnType(i), [&](auto tag) {
      using ColumnT = typename decltype(tag)::type;
      m_fields.push_back(std::make_unique<FlatTableField<ColumnT>>(table, i, model));
    });
  }
}

void TableOutputFields::bind(ROOT::REntry& entry) const {
  for (const auto& field : m_fields) {
    field->bind(entry);
  }
}

void TableOutputFields::fillEntry(const nanoaod::FlatTable& table, std::size_t i) {
  for (auto& field : m_fields) {
    field->fillRow(table, i);
  }
}

const edm::EDGetToken& TableOutputFields::getToken() const { return m_token; }

edm::Handle<nanoaod::FlatTable> TableOutputFields::getTable(const edm::OccurrenceForOutput& event) const {
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_token, handle);
  return handle;
}

///////////////////////////////////////////////////////////////////////////////

void TableOutputVectorFields::createFields(const edm::OccurrenceForOutput& event, RNTupleModel& model) {
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_token, handle);
  const nanoaod::FlatTable& table = *handle;
  m_fields.reserve(table.nColumns());
  for (std::size_t i = 0; i < table.nColumns(); i++) {
    dispatchColumnType(table.columnType(i), [&](auto tag) {
      using ColumnT = typename decltype(tag)::type;
      m_fields.push_back(std::make_unique<FlatTableVectorField<ColumnT>>(table, i, model));
    });
  }
}

void TableOutputVectorFields::bind(ROOT::REntry& entry) const {
  for (const auto& field : m_fields) {
    field->bind(entry);
  }
}

void TableOutputVectorFields::fill(const edm::OccurrenceForOutput& event) {
  edm::Handle<nanoaod::FlatTable> handle;
  event.getByToken(m_token, handle);
  for (auto& field : m_fields) {
    field->fillColumn(*handle);
  }
}

///////////////////////////////////////////////////////////////////////////////

void TableCollection::add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table) {
  if (m_collectionName.empty()) {
    m_collectionName = table.name();
  }
  if (table.extension()) {
    m_extensions.emplace_back(table_token);
    return;
  }
  if (hasMainTable()) {
    throw cms::Exception("LogicError", "Trying to save multiple main tables for " + m_collectionName + "\n");
  }
  m_singleton = table.singleton();
  m_main = TableOutputFields(table_token);
}

std::vector<edm::Handle<nanoaod::FlatTable>> TableCollection::getTables(const edm::OccurrenceForOutput& event) const {
  std::vector<edm::Handle<nanoaod::FlatTable>> tables;
  tables.reserve(m_extensions.size() + 1);
  tables.emplace_back(m_main.getTable(event));
  for (const auto& extension : m_extensions) {
    tables.emplace_back(extension.getTable(event));
  }
  return tables;
}

void TableCollection::createFields(const edm::OccurrenceForOutput& event, RNTupleModel& eventModel) {
  auto tables = getTables(event);
  m_collection =
      std::make_unique<RNTupleCollection>(m_collectionName, tables.front()->doc(), tables, eventModel, m_singleton);
}

void TableCollection::addProjections(RNTupleModel& eventModel) const { m_collection->addProjections(eventModel); }

void TableCollection::bind(ROOT::REntry& entry) const { m_collection->bindBuffer(entry); }

void TableCollection::fill(const edm::OccurrenceForOutput& event) {
  auto tables = getTables(event);
  m_collection->fill(tables);
}

bool TableCollection::hasMainTable() const { return !m_main.getToken().isUninitialized(); }

const std::string& TableCollection::getCollectionName() const { return m_collectionName; }

///////////////////////////////////////////////////////////////////////////////

void TableCollectionSet::add(const edm::EDGetToken& table_token, const nanoaod::FlatTable& table) {
  // skip empty tables -- requirement of RNTuple to define schema before filling
  if (table.nColumns() == 0) {
    edm::LogWarning("TableCollectionSet") << "Skipping FlatTable '" << table.name() << "': it has no columns\n";
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
      m_singletonFields.emplace_back(table_token);
    } else {
      m_vectorFields.emplace_back(table_token);
    }
    return;
  }
  // case 2: Named singleton and vector tables are both written as RNTuple collections.
  auto collection = std::find_if(m_collections.begin(), m_collections.end(), [&](const TableCollection& c) {
    return c.getCollectionName() == table.name();
  });
  if (collection == m_collections.end()) {
    m_collections.emplace_back();
    m_collections.back().add(table_token, table);
    return;
  }
  collection->add(table_token, table);
}

void TableCollectionSet::createFields(const edm::OccurrenceForOutput& event, RNTupleModel& eventModel) {
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

void TableCollectionSet::addProjections(RNTupleModel& eventModel) const {
  for (const auto& collection : m_collections) {
    collection.addProjections(eventModel);
  }
}

void TableCollectionSet::bind(ROOT::REntry& entry) const {
  for (const auto& collection : m_collections) {
    collection.bind(entry);
  }
  for (const auto& fields : m_singletonFields) {
    fields.bind(entry);
  }
  for (const auto& fields : m_vectorFields) {
    fields.bind(entry);
  }
}

void TableCollectionSet::fill(const edm::OccurrenceForOutput& event) {
  for (auto& collection : m_collections) {
    collection.fill(event);
  }
  for (auto& fields : m_singletonFields) {
    fields.fillEntry(*fields.getTable(event), 0);
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
                         "FlatTable `" + table.name() + "` has " + std::to_string(num_anon) + " anonymous fields");
  }
  return num_anon > 0;
}
