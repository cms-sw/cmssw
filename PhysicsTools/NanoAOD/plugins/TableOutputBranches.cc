#include "PhysicsTools/NanoAOD/plugins/TableOutputBranches.h"

#include <iostream>
#include <limits>

namespace {
  std::string makeBranchName(const std::string &baseName, const std::string &leafName) {
    return baseName.empty() ? leafName : (leafName.empty() ? baseName : baseName + "_" + leafName);
  }
}  // namespace

void TableOutputBranches::defineBranchesFromFirstEvent(const nanoaod::FlatTable &tab) {
  m_baseName = tab.name();
  for (size_t i = 0; i < tab.nColumns(); i++) {
    const std::string &var = tab.columnName(i);
    switch (tab.columnType(i)) {
      case nanoaod::FlatTable::ColumnType::Int8:
        m_int8Branches.emplace_back(var, tab.columnDoc(i), "B");
        break;
      case nanoaod::FlatTable::ColumnType::UInt8:
        m_uint8Branches.emplace_back(var, tab.columnDoc(i), "b");
        break;
      case nanoaod::FlatTable::ColumnType::Int16:
        m_int16Branches.emplace_back(var, tab.columnDoc(i), "S");
        break;
      case nanoaod::FlatTable::ColumnType::UInt16:
        m_uint16Branches.emplace_back(var, tab.columnDoc(i), "s");
        break;
      case nanoaod::FlatTable::ColumnType::Int32:
        m_int32Branches.emplace_back(var, tab.columnDoc(i), "I");
        break;
      case nanoaod::FlatTable::ColumnType::UInt32:
        m_uint32Branches.emplace_back(var, tab.columnDoc(i), "i");
        break;
      case nanoaod::FlatTable::ColumnType::Bool:
        m_uint8Branches.emplace_back(var, tab.columnDoc(i), "O");
        break;
      case nanoaod::FlatTable::ColumnType::Float:
        m_floatBranches.emplace_back(var, tab.columnDoc(i), "F");
        break;
      case nanoaod::FlatTable::ColumnType::Double:
        m_doubleBranches.emplace_back(var, tab.columnDoc(i), "D");
        break;
      default:
        throw cms::Exception("LogicError", "Unsupported type");
    }
  }
}

void TableOutputBranches::branch(TTree &tree) {
  if (!m_singleton) {
    if (m_extension == IsExtension) {
      m_counterBranch = tree.FindBranch(("n" + m_baseName).c_str());
      if (!m_counterBranch) {
        throw cms::Exception("LogicError",
                             "Trying to save an extension table for " + m_baseName +
                                 " before having saved the corresponding main table\n");
      }
    } else {
      if (tree.FindBranch(("n" + m_baseName).c_str()) != nullptr) {
        throw cms::Exception("LogicError", "Trying to save multiple main tables for " + m_baseName + "\n");
      }
      m_counterBranch = tree.Branch(("n" + m_baseName).c_str(), &m_counter, ("n" + m_baseName + "/I").c_str());
      m_counterBranch->SetTitle(m_doc.c_str());
    }
  }
  std::string varsize = m_singleton ? "" : "[n" + m_baseName + "]";
  for (std::vector<NamedBranchPtr> *branches : {&m_int8Branches,
                                                &m_uint8Branches,
                                                &m_int16Branches,
                                                &m_uint16Branches,
                                                &m_int32Branches,
                                                &m_uint32Branches,
                                                &m_floatBranches,
                                                &m_doubleBranches}) {
    for (auto &pair : *branches) {
      std::string branchName = makeBranchName(m_baseName, pair.name);
      pair.branch =
          tree.Branch(branchName.c_str(), (void *)nullptr, (branchName + varsize + "/" + pair.rootTypeCode).c_str());
      pair.branch->SetTitle(pair.title.c_str());
    }
  }
}

void TableOutputBranches::fill(const edm::OccurrenceForOutput &iWhatever, TTree &tree, bool extensions) {
  if (m_extension != DontKnowYetIfMainOrExtension) {
    if (extensions != m_extension)
      return;  // do nothing, wait to be called with the proper flag
  }

  edm::Handle<nanoaod::FlatTable> handle;
  iWhatever.getByToken(m_token, handle);
  const nanoaod::FlatTable &tab = *handle;
  auto size = tab.size();
  // ROOT native array size branches may only be signed integers,
  // until this is changed we need to make sure the vector sizes do not exceed that
  if (size > std::numeric_limits<CounterType>::max()) {
    throw cms::Exception("Table " + tab.name() + " size is " + std::to_string(size) +
                         ", is too large for ROOT native array branch");
  }
  m_counter = size;
  m_singleton = tab.singleton();
  if (!m_branchesBooked) {
    m_extension = tab.extension() ? IsExtension : IsMain;
    if (extensions != m_extension)
      return;  // do nothing, wait to be called with the proper flag
    defineBranchesFromFirstEvent(tab);
    m_doc = tab.doc();
    m_branchesBooked = true;
    branch(tree);
  }
  if (!m_singleton && m_extension == IsExtension) {
    if (m_counter != *reinterpret_cast<CounterType *>(m_counterBranch->GetAddress())) {
      throw cms::Exception("LogicError",
                           "Mismatch in number of entries between extension and main table for " + tab.name());
    }
  }
  for (auto &pair : m_int8Branches)
    fillColumn<int8_t>(pair, tab);
  for (auto &pair : m_uint8Branches)
    fillColumn<uint8_t>(pair, tab);
  for (auto &pair : m_int16Branches)
    fillColumn<int16_t>(pair, tab);
  for (auto &pair : m_uint16Branches)
    fillColumn<uint16_t>(pair, tab);
  for (auto &pair : m_int32Branches)
    fillColumn<int32_t>(pair, tab);
  for (auto &pair : m_uint32Branches)
    fillColumn<uint32_t>(pair, tab);
  for (auto &pair : m_floatBranches)
    fillColumn<float>(pair, tab);
  for (auto &pair : m_doubleBranches)
    fillColumn<double>(pair, tab);
}
