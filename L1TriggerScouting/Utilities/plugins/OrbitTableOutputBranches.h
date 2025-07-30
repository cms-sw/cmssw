#ifndef L1TriggerScouting_Utilities_OrbitTableOutputBranches_h
#define L1TriggerScouting_Utilities_OrbitTableOutputBranches_h

#include <string>
#include <vector>
#include <TTree.h>
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "DataFormats/NanoAOD/interface/OrbitFlatTable.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class OrbitTableOutputBranches {
public:
  OrbitTableOutputBranches(const edm::BranchDescription *desc, const edm::EDGetToken &token)
      : m_token(token), m_extension(DontKnowYetIfMainOrExtension), m_branchesBooked(false) {
    if (desc->className() != "l1ScoutingRun3::OrbitFlatTable")
      throw cms::Exception("Configuration",
                           "OrbitNanoAODOutputModule can only write out l1ScoutingRun3::OrbitFlatTable objects");
  }

  void defineBranchesFromFirstEvent(const l1ScoutingRun3::OrbitFlatTable &tab);
  void branch(TTree &tree);

  /// Fill the current table, if extensions == table.extension().
  /// This parameter is used so that the fill is called first for non-extensions and then for extensions
  void beginFill(const edm::OccurrenceForOutput &iWhatever, TTree &tree, bool extensions);
  bool hasBx(uint32_t bx);
  void fillBx(uint32_t bx);
  void endFill();

private:
  edm::EDGetToken m_token;
  std::string m_baseName;
  bool m_singleton = false;
  enum { IsMain = 0, IsExtension = 1, DontKnowYetIfMainOrExtension = 2 } m_extension;
  std::string m_doc;
  typedef Int_t CounterType;
  CounterType m_counter;
  struct NamedBranchPtr {
    std::string name, title, rootTypeCode;
    int columnIndex;
    TBranch *branch;
    NamedBranchPtr(const std::string &aname,
                   const std::string &atitle,
                   const std::string &rootType,
                   int columnIndex,
                   TBranch *branchptr = nullptr)
        : name(aname), title(atitle), rootTypeCode(rootType), columnIndex(columnIndex), branch(branchptr) {}
  };
  TBranch *m_counterBranch = nullptr;
  std::vector<NamedBranchPtr> m_uint8Branches;
  std::vector<NamedBranchPtr> m_int16Branches;
  std::vector<NamedBranchPtr> m_uint16Branches;
  std::vector<NamedBranchPtr> m_int32Branches;
  std::vector<NamedBranchPtr> m_uint32Branches;
  std::vector<NamedBranchPtr> m_floatBranches;
  std::vector<NamedBranchPtr> m_doubleBranches;
  bool m_branchesBooked;

  edm::Handle<l1ScoutingRun3::OrbitFlatTable> m_handle;
  const l1ScoutingRun3::OrbitFlatTable *m_table;

  template <typename T>
  void fillColumn(NamedBranchPtr &pair, uint32_t bx) {
    pair.branch->SetAddress(
        m_counter == 0
            ? static_cast<T *>(nullptr)
            : const_cast<T *>(
                  &m_table->columnData<T>(pair.columnIndex, bx).front()));  // SetAddress should take a const * !
  }
};

#endif
