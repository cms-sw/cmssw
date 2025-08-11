#include "L1TriggerScouting/Utilities/plugins/OrbitTableOutputBranches.h"

#include <limits>

namespace {
  std::string makeBranchName(const std::string &baseName, const std::string &leafName) {
    return baseName.empty() ? leafName : (leafName.empty() ? baseName : baseName + "_" + leafName);
  }
}  // namespace

void OrbitTableOutputBranches::defineBranchesFromFirstEvent(const l1ScoutingRun3::OrbitFlatTable &tab) {
  m_baseName = tab.name();
  for (size_t i = 0; i < tab.nColumns(); i++) {
    const std::string &var = tab.columnName(i);
    // get columnIndex
    int columnIndex = tab.columnIndex(var);
    if (columnIndex == -1)
      throw cms::Exception("LogicError", "Missing column in input for " + m_baseName + "_" + var);

    switch (tab.columnType(i)) {
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::UInt8:
        m_uint8Branches.emplace_back(var, tab.columnDoc(i), "b", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::Int16:
        m_int16Branches.emplace_back(var, tab.columnDoc(i), "S", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::UInt16:
        m_uint16Branches.emplace_back(var, tab.columnDoc(i), "s", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::Int32:
        m_int32Branches.emplace_back(var, tab.columnDoc(i), "I", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::UInt32:
        m_uint32Branches.emplace_back(var, tab.columnDoc(i), "i", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::Bool:
        m_uint8Branches.emplace_back(var, tab.columnDoc(i), "O", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::Float:
        m_floatBranches.emplace_back(var, tab.columnDoc(i), "F", columnIndex);
        break;
      case l1ScoutingRun3::OrbitFlatTable::ColumnType::Double:
        m_doubleBranches.emplace_back(var, tab.columnDoc(i), "D", columnIndex);
        break;
      default:
        throw cms::Exception("LogicError", "Unsupported type");
    }
  }
}

void OrbitTableOutputBranches::branch(TTree &tree) {
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
  for (std::vector<NamedBranchPtr> *branches : {&m_uint8Branches,
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

void OrbitTableOutputBranches::beginFill(const edm::OccurrenceForOutput &iWhatever, TTree &tree, bool extensions) {
  if (m_extension != DontKnowYetIfMainOrExtension) {
    if (extensions != m_extension)
      return;  // do nothing, wait to be called with the proper flag
  }

  iWhatever.getByToken(m_token, m_handle);
  m_table = m_handle.product();
  m_singleton = m_table->singleton();
  if (!m_branchesBooked) {
    m_extension = m_table->extension() ? IsExtension : IsMain;
    if (extensions != m_extension)
      return;  // do nothing, wait to be called with the proper flag
    defineBranchesFromFirstEvent(*m_table);
    m_doc = m_table->doc();
    m_branchesBooked = true;
    branch(tree);
  }
}

bool OrbitTableOutputBranches::hasBx(uint32_t bx) { return (m_table->size(bx) != 0); }

void OrbitTableOutputBranches::fillBx(uint32_t bx) {
  m_counter = m_table->size(bx);
  if (m_counter != 0) {
    for (auto &pair : m_uint8Branches)
      fillColumn<uint8_t>(pair, bx);
    for (auto &pair : m_int16Branches)
      fillColumn<int16_t>(pair, bx);
    for (auto &pair : m_uint16Branches)
      fillColumn<uint16_t>(pair, bx);
    for (auto &pair : m_int32Branches)
      fillColumn<int32_t>(pair, bx);
    for (auto &pair : m_uint32Branches)
      fillColumn<uint32_t>(pair, bx);
    for (auto &pair : m_floatBranches)
      fillColumn<float>(pair, bx);
    for (auto &pair : m_doubleBranches)
      fillColumn<double>(pair, bx);
  }
}

void OrbitTableOutputBranches::endFill() { m_table = nullptr; }
