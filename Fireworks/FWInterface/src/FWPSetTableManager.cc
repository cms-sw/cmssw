// -*- C++ -*-
//
// Package:     FWInterface
// Class  :     FWPSetTableManager
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:
//         Created:  Mon Feb 28 17:06:54 CET 2011
//

#include <map>
#include <stdexcept>

#include "Fireworks/FWInterface/src/FWPSetTableManager.h"
#include "Fireworks/FWInterface/src/FWPSetCellEditor.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/TableWidget/interface/GlobalContexts.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "FWCore/Framework/interface/ScheduleInfo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

// FIXME: copied from Entry.cc should find a way to use the original
//        table.
struct TypeTrans {
  TypeTrans();

  typedef std::vector<std::string> CodeMap;
  CodeMap table_;
  std::map<std::string, char> type2Code_;
};

TypeTrans::TypeTrans() : table_(255) {
  table_['b'] = "vBool";
  table_['B'] = "bool";
  table_['i'] = "vint32";
  table_['I'] = "int32";
  table_['u'] = "vuint32";
  table_['U'] = "uint32";
  table_['l'] = "vint64";
  table_['L'] = "int64";
  table_['x'] = "vuint64";
  table_['X'] = "uint64";
  table_['s'] = "vstring";
  table_['S'] = "string";
  table_['d'] = "vdouble";
  table_['D'] = "double";
  table_['p'] = "vPSet";
  table_['P'] = "PSet";
  table_['T'] = "path";
  table_['F'] = "FileInPath";
  table_['t'] = "InputTag";
  table_['v'] = "VInputTag";
  table_['g'] = "ESInputTag";
  table_['G'] = "VESInputTag";
  table_['e'] = "VEventID";
  table_['E'] = "EventID";
  table_['m'] = "VLuminosityBlockID";
  table_['M'] = "LuminosityBlockID";
  table_['a'] = "VLuminosityBlockRange";
  table_['A'] = "LuminosityBlockRange";
  table_['r'] = "VEventRange";
  table_['R'] = "EventRange";

  for (CodeMap::const_iterator itCode = table_.begin(), itCodeEnd = table_.end(); itCode != itCodeEnd; ++itCode) {
    type2Code_[*itCode] = (itCode - table_.begin());
  }
}

static TypeTrans const sTypeTranslations;

//
// constructors and destructor
//

FWPSetTableManager::FWPSetTableManager() : m_selectedRow(-1) {
  TGGC *hc = new TGGC(FWTextTableCellRenderer::getDefaultHighlightGC());
  hc->SetForeground(0xdddddd);

  m_renderer.setHighlightContext(hc);

  recalculateVisibility();
  visualPropertiesChanged();
}

FWPSetTableManager::~FWPSetTableManager() {}

//==============================================================================
//==============================================================================
//========== IMPORT CMSSW CONFIG TO TABLE ======================================
//==============================================================================
//==============================================================================

void FWPSetTableManager::handlePSetEntry(edm::ParameterSetEntry &entry, const std::string &key) {
  PSetData data;
  data.label = key;
  data.tracked = entry.isTracked();
  data.level = m_parentStack.size();
  data.parent = m_parentStack.back();
  data.type = 'P';
  data.module = m_modules.size() - 1;
  data.path = m_paths.size() - 1;
  data.pset = &entry.psetForUpdate();
  data.editable = false;
  m_parentStack.push_back(m_entries.size());
  m_entries.push_back(data);

  handlePSet(data.pset);
  m_parentStack.pop_back();
}

void FWPSetTableManager::handleVPSetEntry(edm::VParameterSetEntry &entry, const std::string &key) {
  PSetData data;
  data.label = key;
  data.tracked = entry.isTracked();
  data.level = m_parentStack.size();
  data.parent = m_parentStack.back();
  data.type = 'p';
  data.module = m_modules.size() - 1;
  data.path = m_paths.size() - 1;
  data.editable = false;
  m_parentStack.push_back(m_entries.size());
  m_entries.push_back(data);

  std::stringstream ss;

  for (size_t i = 0, e = entry.vpset().size(); i != e; ++i) {
    ss.str("");
    ss << key << "[" << i << "]";
    PSetData vdata;
    vdata.label = ss.str();
    vdata.tracked = entry.isTracked();
    vdata.level = m_parentStack.size();
    vdata.parent = m_parentStack.back();
    vdata.module = m_modules.size() - 1;
    vdata.path = m_paths.size() - 1;
    vdata.editable = false;
    vdata.pset = &entry.vpsetForUpdate()[i];
    m_parentStack.push_back(m_entries.size());
    m_entries.push_back(vdata);
    handlePSet(&entry.vpsetForUpdate()[i]);
    m_parentStack.pop_back();
  }
  m_parentStack.pop_back();
}

void FWPSetTableManager::handlePSet(edm::ParameterSet *psp) {
  edm::ParameterSet &ps = *psp;

  typedef edm::ParameterSet::table::const_iterator TIterator;
  for (TIterator i = ps.tbl().begin(), e = ps.tbl().end(); i != e; ++i)
    handleEntry(i->second, i->first);

  typedef edm::ParameterSet::psettable::const_iterator PSIterator;
  for (PSIterator i = ps.psetTable().begin(), e = ps.psetTable().end(); i != e; ++i)
    handlePSetEntry(const_cast<edm::ParameterSetEntry &>(i->second), i->first);

  typedef edm::ParameterSet::vpsettable::const_iterator VPSIterator;
  for (VPSIterator i = ps.vpsetTable().begin(), e = ps.vpsetTable().end(); i != e; ++i)
    handleVPSetEntry(const_cast<edm::VParameterSetEntry &>(i->second), i->first);
}

template <class T>
void FWPSetTableManager::createScalarString(PSetData &data, T v) {
  std::stringstream ss;
  ss << v;
  data.value = ss.str();
  m_entries.push_back(data);
}

template <typename T>
void FWPSetTableManager::createVectorString(FWPSetTableManager::PSetData &data, const T &v, bool quotes) {
  std::stringstream ss;
  ss << "[";
  for (size_t ii = 0, ie = v.size(); ii != ie; ++ii) {
    if (quotes)
      ss << "\"";
    ss << v[ii];
    if (quotes)
      ss << "\"";
    if (ii + 1 != ie)
      ss << ", ";
  }
  ss << "]";
  data.value = ss.str();
  m_entries.push_back(data);
}

void FWPSetTableManager::handleEntry(const edm::Entry &entry, const std::string &key) {
  std::stringstream ss;
  FWPSetTableManager::PSetData data;
  data.label = key;
  data.tracked = entry.isTracked();
  data.type = entry.typeCode();
  data.level = m_parentStack.size();
  data.parent = m_parentStack.back();
  data.module = m_modules.size() - 1;
  data.type = entry.typeCode();
  if (data.label[0] == '@')
    data.editable = false;
  else
    data.editable = true;

  switch (entry.typeCode()) {
    case 'b': {
      data.value = entry.getBool() ? "True" : "False";
      m_entries.push_back(data);
      break;
    }
    case 'B': {
      data.value = entry.getBool() ? "True" : "False";
      m_entries.push_back(data);
      break;
    }
    case 'i': {
      createVectorString(data, entry.getVInt32(), false);
      break;
    }
    case 'I': {
      createScalarString(data, entry.getInt32());
      break;
    }
    case 'u': {
      createVectorString(data, entry.getVUInt32(), false);
      break;
    }
    case 'U': {
      createScalarString(data, entry.getUInt32());
      break;
    }
    case 'l': {
      createVectorString(data, entry.getVInt64(), false);
      break;
    }
    case 'L': {
      createScalarString(data, entry.getInt32());
      break;
    }
    case 'x': {
      createVectorString(data, entry.getVUInt64(), false);
      break;
    }
    case 'X': {
      createScalarString(data, entry.getUInt64());
      break;
    }
    case 's': {
      createVectorString(data, entry.getVString(), false);
      break;
    }
    case 'S': {
      createScalarString(data, entry.getString());
      break;
    }
    case 'd': {
      createVectorString(data, entry.getVDouble(), false);
      break;
    }
    case 'D': {
      createScalarString(data, entry.getDouble());
      break;
    }
    case 'p': {
      // Matevz ???
      throw std::runtime_error("FWPSetTableManager::handleEntryGet, entry type 'p' not expected.");
      // std::vector<edm::ParameterSet> psets = entry.getVPSet();
      // for (size_t psi = 0, pse = psets.size(); psi != pse; ++psi)
      //    handlePSet(psets[psi]);
      break;
    }
    case 'P': {
      // Matevz ???
      throw std::runtime_error("FWPSetTableManager::handleEntry, entry type 'P not expected.");
      // handlePSet(entry.getPSet());
      break;
    }
    case 't': {
      data.value = entry.getInputTag().encode();
      m_entries.push_back(data);
      break;
    }
    case 'v': {
      std::vector<std::string> tags;
      tags.resize(entry.getVInputTag().size());
      for (size_t iti = 0, ite = tags.size(); iti != ite; ++iti)
        tags[iti] = entry.getVInputTag()[iti].encode();
      createVectorString(data, tags, true);
      break;
    }
    case 'g': {
      data.value = entry.getESInputTag().encode();
      m_entries.push_back(data);
      break;
    }
    case 'G': {
      std::vector<std::string> tags;
      tags.resize(entry.getVESInputTag().size());
      for (size_t iti = 0, ite = tags.size(); iti != ite; ++iti)
        tags[iti] = entry.getVESInputTag()[iti].encode();
      createVectorString(data, tags, true);
      break;
    }
    case 'F': {
      createScalarString(data, entry.getFileInPath().relativePath());
      break;
    }
    case 'e': {
      data.editable = false;
      std::vector<edm::EventID> ids;
      ids.resize(entry.getVEventID().size());
      for (size_t iri = 0, ire = ids.size(); iri != ire; ++iri)
        ids[iri] = entry.getVEventID()[iri];
      createVectorString(data, ids, true);
      break;
    }
    case 'E': {
      data.editable = false;
      createScalarString(data, entry.getEventID());
      break;
    }
    case 'm': {
      data.editable = false;
      std::vector<edm::LuminosityBlockID> ids;
      ids.resize(entry.getVLuminosityBlockID().size());
      for (size_t iri = 0, ire = ids.size(); iri != ire; ++iri)
        ids[iri] = entry.getVLuminosityBlockID()[iri];
      createVectorString(data, ids, true);
      break;
    }
    case 'M': {
      data.editable = false;
      createScalarString(data, entry.getLuminosityBlockID());
      break;
    }
    case 'a': {
      data.editable = false;
      std::vector<edm::LuminosityBlockRange> ranges;
      ranges.resize(entry.getVLuminosityBlockRange().size());
      for (size_t iri = 0, ire = ranges.size(); iri != ire; ++iri)
        ranges[iri] = entry.getVLuminosityBlockRange()[iri];
      createVectorString(data, ranges, true);
      break;
    }
    case 'A': {
      data.editable = false;
      createScalarString(data, entry.getLuminosityBlockRange());
      break;
    }
    case 'r': {
      data.editable = false;
      std::vector<edm::EventRange> ranges;
      ranges.resize(entry.getVEventRange().size());
      for (size_t iri = 0, ire = ranges.size(); iri != ire; ++iri)
        ranges[iri] = entry.getVEventRange()[iri];
      createVectorString(data, ranges, true);
      break;
    }
    case 'R': {
      data.editable = false;
      createScalarString(data, entry.getEventRange());
      break;
    }
    default: {
      break;
    }
  }
}

/* the actual structure of the model will not change, only
   its contents, because of the way CMSSW is designed,
   hence this method only needs to be called once.
   */
void FWPSetTableManager::updateSchedule(const edm::ScheduleInfo *info) {
  if (!m_entries.empty())
    return;
  // Execute only once since the schedule itself
  // cannot be altered.
  assert(m_availablePaths.empty());
  info->availablePaths(m_availablePaths);

  for (size_t i = 0, e = m_availablePaths.size(); i != e; ++i) {
    PSetData pathEntry;
    const std::string &pathName = m_availablePaths[i];
    pathEntry.label = pathName;
    m_pathIndex.insert(std::make_pair(pathName, m_paths.size()));

    pathEntry.value = "Path";
    pathEntry.level = 0;
    pathEntry.parent = -1;
    pathEntry.path = i;
    pathEntry.editable = false;

    PathInfo pathInfo;
    pathInfo.entryId = m_entries.size();
    pathInfo.passed = false;
    pathInfo.moduleStart = m_modules.size();
    m_paths.push_back(pathInfo);

    m_parentStack.push_back(m_entries.size());
    m_entries.push_back(pathEntry);

    std::vector<std::string> pathModules;
    info->modulesInPath(pathName, pathModules);

    for (size_t mi = 0, me = pathModules.size(); mi != me; ++mi) {
      PSetData moduleEntry;

      const edm::ParameterSet *ps = info->parametersForModule(pathModules[mi]);

      const edm::ParameterSet::table &pst = ps->tbl();
      const edm::ParameterSet::table::const_iterator ti = pst.find("@module_edm_type");
      if (ti == pst.end())
        moduleEntry.value = "Unknown module name";
      else
        moduleEntry.value = ti->second.getString();

      moduleEntry.label = pathModules[mi];
      moduleEntry.parent = m_parentStack.back();
      moduleEntry.level = m_parentStack.size();
      moduleEntry.module = mi;
      moduleEntry.path = i;
      moduleEntry.editable = false;

      ModuleInfo moduleInfo;
      moduleInfo.path = m_paths.size() - 1;
      moduleInfo.entry = m_entries.size();
      moduleInfo.passed = false;
      moduleInfo.dirty = false;
      moduleInfo.orig_pset = new edm::ParameterSet(*ps);
      moduleInfo.current_pset = new edm::ParameterSet(*ps);
      m_modules.push_back(moduleInfo);

      moduleEntry.pset = moduleInfo.current_pset;

      m_parentStack.push_back(m_entries.size());
      m_entries.push_back(moduleEntry);
      handlePSet(moduleEntry.pset);
      m_parentStack.pop_back();
    }
    m_paths.back().moduleEnd = m_modules.size();
    m_parentStack.pop_back();
  }

  // Nothing is expanded by default.
  for (size_t i = 0, e = m_entries.size(); i != e; ++i)
    m_entries[i].expandedUser = false;

  m_filter = "";

  recalculateVisibility();
}  //updateSchedule

/** Update the status of a given path. This is the information 
    that changes on event by event basis.
   */
void FWPSetTableManager::update(std::vector<PathUpdate> &pathUpdates) {
  // Reset all the path / module status information, so that
  // by default paths and modules are considered "not passed".
  for (size_t pi = 0, pe = m_paths.size(); pi != pe; ++pi)
    m_paths[pi].passed = false;
  for (size_t mi = 0, me = m_modules.size(); mi != me; ++mi)
    m_modules[mi].passed = false;

  // Update whether or not a given path / module passed selection.
  for (size_t pui = 0, pue = pathUpdates.size(); pui != pue; ++pui) {
    PathUpdate &update = pathUpdates[pui];
    std::map<std::string, size_t>::const_iterator index = m_pathIndex.find(update.pathName);
    if (index == m_pathIndex.end()) {
      fwLog(fwlog::kError) << "Path " << update.pathName << "cannot be found!" << std::endl;
      continue;
    }
    PathInfo &pathInfo = m_paths[index->second];
    pathInfo.passed = update.passed;

    for (size_t mi = pathInfo.moduleStart, me = pathInfo.moduleEnd; mi != me; ++mi) {
      ModuleInfo &moduleInfo = m_modules[mi];
      moduleInfo.passed = update.passed || ((mi - pathInfo.moduleStart) < update.choiceMaker);
    }
  }

  implSort(-1, true);
}

//==============================================================================
//==============================================================================
//=============== CELL EDITOR ACTIONS ==========================================
//==============================================================================
//==============================================================================

void FWPSetTableManager::setCellValueEditor(FWPSetCellEditor *editor) {
  m_editor = editor;
  m_renderer.setCellEditor(m_editor);
}

/** Does not apply changes and closes window. */
void FWPSetTableManager::cancelEditor() {
  if (!m_editor)
    return;

  //  printf("FWPSetTableManager::cancelEditor() \n");
  setSelection(-1, -1, 0);
  m_editor->UnmapWindow();
}

/** This is invoked every single time the
       editor contents must be applied to the selected entry in the pset. 
       @return true on success. 
*/
bool FWPSetTableManager::applyEditor() {
  if (!m_editor)
    return false;

  if (m_selectedRow == -1 || m_selectedColumn != 1)
    return false;

  //  printf("FWPSetTableManager::applyEditor() \n");
  PSetData &data = m_entries[m_row_to_index[m_selectedRow]];
  PSetData &parent = m_entries[data.parent];
  bool success = false;
  try {
    success = m_editor->apply(data, parent);

    if (success) {
      data.value = m_editor->GetText();
      m_modules[data.module].dirty = true;
      setSelection(-1, -1, 0);
      m_editor->UnmapWindow();
      // ???
      // copy current to orig
    } else {
      // ???
      // set current from orig? reimport module ... hmmh, hard.
    }
  } catch (cms::Exception &e) {
    m_editor->SetForegroundColor(gVirtualX->GetPixel(kRed));
  }
  return success;
}

//==============================================================================
//==============================================================================
//========  TABLE UI MNG (virutals FWTableManagerBase virtuals, etc.)   =========
//==============================================================================
//==============================================================================
const std::string FWPSetTableManager::title() const { return "Modules & their parameters"; }

std::vector<std::string> FWPSetTableManager::getTitles() const {
  std::vector<std::string> returnValue;
  returnValue.reserve(numberOfColumns());
  returnValue.push_back("Label");
  returnValue.push_back("Value");
  return returnValue;
}

int FWPSetTableManager::selectedRow() const { return m_selectedRow; }

int FWPSetTableManager::selectedColumn() const { return m_selectedColumn; }

bool FWPSetTableManager::rowIsSelected(int row) const { return m_selectedRow == row; }

int FWPSetTableManager::unsortedRowNumber(int unsorted) const { return unsorted; }

int FWPSetTableManager::numberOfRows() const { return m_row_to_index.size(); }

int FWPSetTableManager::numberOfColumns() const { return 2; }

void FWPSetTableManager::setSelection(int iRow, int iColumn, int mask) {
  // printf("set selection %d %d mode %d\n", iRow, iColumn, mask);

  // Nothing changes if we clicked selected
  // twice the same cell.
  if (iRow == m_selectedRow && iColumn == m_selectedColumn)
    return;

  // Otherwise update the selection information
  // and notify observers.
  m_selectedRow = iRow;
  m_selectedColumn = iColumn;
  if (iColumn == 1 && iRow > 0) {
    int unsortedRow = m_row_to_index[iRow];
    const PSetData &data = m_entries[unsortedRow];
    if (m_editor && data.editable) {
      m_editor->MoveResize(0, cellHeight() * iRow, m_editor->GetWidth(), m_editor->GetHeight());
      m_editor->MapWindow();
      m_editor->SetText(data.value.c_str());
      m_editor->SetFocus();
      m_editor->SetCursorPosition(data.value.size() - 1);
    }
  } else {
    if (m_editor)
      m_editor->UnmapWindow();
  }
  visualPropertiesChanged();
}

std::vector<unsigned int> FWPSetTableManager::maxWidthForColumns() const {
  std::vector<unsigned int> ww = FWTableManagerBase::maxWidthForColumns();
  if (ww.size() > 1 && ww[1] > 0) {
    // printf("dim W %d \n",ww[1]);
    // printf("dim H %d \n",cellHeight());
    if (m_editor)
      m_editor->MoveResize(m_editor->GetX(), m_editor->GetY(), ww[1], cellHeight());
  }
  return ww;
}

void FWPSetTableManager::implSort(int, bool) {}
//______________________________________________________________________________

void FWPSetTableManager::setExpanded(int row) {
  if (row == -1)
    return;

  int index = rowToIndex()[row];
  PSetData &data = m_entries[index];

  if (m_filter.empty() == false && data.childMatches == false)
    return;

  if (m_filter.empty())
    data.expandedUser = !data.expandedUser;
  else
    data.expandedFilter = !data.expandedFilter;

  recalculateVisibility();
  dataChanged();
  visualPropertiesChanged();
}

//______________________________________________________________________________

FWTableCellRendererBase *FWPSetTableManager::cellRenderer(int iSortedRowNumber, int iCol) const {
  const static size_t maxSize = 512;  // maximum string length

  static TGGC boldGC(fireworks::boldGC());
  static TGGC italicGC(fireworks::italicGC());
  static TGGC defaultGC(FWTextTableCellRenderer::getDefaultGC());

  const static Pixel_t gray = 0x777777;
  const static Pixel_t red = gVirtualX->GetPixel(kRed - 5);
  const static Pixel_t green = gVirtualX->GetPixel(kGreen - 5);

  // return in case if nothing maches filter
  if (static_cast<int>(m_row_to_index.size()) <= iSortedRowNumber) {
    m_renderer.setData(std::string(), false);
    return &m_renderer;
  }

  int unsortedRow = m_row_to_index[iSortedRowNumber];
  const PSetData &data = m_entries[unsortedRow];

  std::string value;
  std::string label;
  TGGC *gc = nullptr;
  if (data.level == 0) {
    const PathInfo &path = m_paths[data.path];
    label = data.label + " (" + data.value + ")";
    gc = &boldGC;
    gc->SetForeground(path.passed ? green : red);
  } else if (data.level == 1) {
    // "passed" means if module made decision on path
    const ModuleInfo &module = m_modules[m_paths[data.path].moduleStart + data.module];
    label = data.label + " (" + data.value + ")";
    gc = (TGGC *)&boldGC;
    gc->SetForeground(module.passed ? green : red);
  } else {
    if (data.type > 0)
      label = data.label + " (" + sTypeTranslations.table_[data.type] + ")";
    else
      label = data.label;
    value = data.value;

    if (data.editable) {
      gc = &defaultGC;
    } else {
      gc = &italicGC;
      gc->SetForeground(gray);
    }
  }

  // check string size and cut it if necessary (problems with X11)
  if (iCol == 1 && value.size() >= maxSize) {
    if (iSortedRowNumber == m_selectedRow)
      fwLog(fwlog::kWarning) << "label: " << label << " has too long value " << value << std::endl << std::endl;

    value = value.substr(0, maxSize);
    value += "[truncated]";
    gc->SetForeground(gVirtualX->GetPixel(kMagenta));
  }

  // debug
  // label = Form("%s m[%d] childm[%d] ", label.c_str(), data.matches, data.childMatches);

  // set text attributes
  m_renderer.setGraphicsContext(gc);
  bool selected = data.matches && (m_filter.empty() == false);
  m_renderer.setData(iCol ? value : label, selected);

  // set  tree attributes
  bool isParent = false;
  bool isOpen = false;
  int indent = 0;
  if (iCol == 0) {
    if (m_filter.empty()) {
      size_t nextIdx = unsortedRow + 1;
      isParent = (nextIdx < m_entries.size() && m_entries[nextIdx].parent == (size_t)unsortedRow);
      isOpen = data.expandedUser;
    } else {
      isParent = data.childMatches;
      isOpen = data.expandedFilter && data.childMatches;
    }

    indent = data.level * 10;
    if (!isParent)
      indent += FWTextTreeCellRenderer::iconWidth();
  }
  m_renderer.setIsParent(isParent);
  m_renderer.setIsOpen(isOpen);
  m_renderer.setIndentation(indent);

  // If we are rendering the selected cell,
  // we show the editor.
  bool showEdit =
      (iCol == 1 && iSortedRowNumber == m_selectedRow && iCol == m_selectedColumn && value.size() < maxSize);
  m_renderer.showEditor(data.editable && showEdit);

  return &m_renderer;
}  // cellRender()

//______________________________________________________________________________

void FWPSetTableManager::updateFilter(const char *filter) {
  m_filter = filter;

  if (m_filter.empty()) {
    // collapse entries when filter is removed
    for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      m_entries[i].expandedFilter = false;
  } else {
    // Decide whether or not items match the filter.
    for (size_t i = 0, e = m_entries.size(); i != e; ++i) {
      PSetData &data = m_entries[i];

      // First of all decide whether or not we match
      // the filter.
      if (strstr(data.label.c_str(), m_filter.c_str()) || strstr(data.value.c_str(), m_filter.c_str()))
        data.matches = true;
      else
        data.matches = false;
    }

    // We reset whether or not a given parent has children that match the
    // filter, and we recompute the whole information by checking all the
    // children.
    for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      m_entries[i].childMatches = false;

    std::vector<int> stack;
    int previousLevel = 0;
    for (size_t i = 0, e = m_entries.size(); i != e; ++i) {
      PSetData &data = m_entries[i];
      // Top level.
      if (data.parent == (size_t)-1) {
        previousLevel = 0;
        // std::cout << "reset stack for top level " << data.label << std::endl;
        stack.clear();
        continue;
      }
      // If the level is greater than the previous one,
      // it means we are among the children of the
      // previous level, hence we push the parent to
      // the stack.
      // If the level is not greater than the previous
      // one it means we have popped out n levels of
      // parents, where N is the difference between the
      // new and the old level. In this case we
      // pop up N parents from the stack.
      if (data.level > previousLevel)
        stack.push_back(data.parent);
      else
        for (size_t pi = 0, pe = previousLevel - data.level; pi != pe; ++pi)
          stack.pop_back();

      if (data.matches && m_entries[stack.back()].childMatches == false) {
        //  printf("match for %s with level %d\n",data.label.c_str(), data.level );
        for (size_t pi = 0, pe = stack.size(); pi != pe; ++pi) {
          //    printf("set child match to parent %s with level %d \n",m_entries[stack[pi]].label.c_str(), m_entries[stack[pi]].level);
          m_entries[stack[pi]].childMatches = true;
        }
      }

      previousLevel = data.level;
    }

    // expand to matching children
    for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      m_entries[i].expandedFilter = m_entries[i].childMatches;
  }

  recalculateVisibility();

  dataChanged();
}  // updateFilter()

//______________________________________________________________________________

void FWPSetTableManager::recalculateVisibility() {
  m_row_to_index.clear();

  // Decide about visibility.
  // * If the items are toplevel and they match the filter, they get shown
  //   in any case.
  // * If the item or any of its children match the filter, the item
  //   is visible.
  // * If the filter is empty and the parent is expanded.
  for (size_t i = 0, e = m_entries.size(); i != e; ++i) {
    PSetData &data = m_entries[i];
    if (data.parent == ((size_t)-1)) {
      data.visible = data.childMatches || data.matches || m_filter.empty();
    } else {
      if (m_filter.empty()) {
        data.visible = m_entries[data.parent].expandedUser && m_entries[data.parent].visible;
      } else {
        if (data.level < 2)
          data.visible = m_entries[data.parent].expandedFilter && m_entries[data.parent].visible &&
                         (data.matches || data.childMatches);
        else
          data.visible = m_entries[data.parent].expandedFilter && m_entries[data.parent].visible;
      }
    }
  }

  // Put in the index only the entries which are visible.
  for (size_t i = 0, e = m_entries.size(); i != e; ++i)
    if (m_entries[i].visible)
      m_row_to_index.push_back(i);
}
