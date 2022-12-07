#ifndef PhysicsTools_NanoAOD_TableOutputBranches_h
#define PhysicsTools_NanoAOD_TableOutputBranches_h

#include <string>
#include <vector>
#include <TTree.h>
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class TableOutputBranches {
public:
  TableOutputBranches(const edm::BranchDescription *desc, const edm::EDGetToken &token)
      : m_token(token), m_extension(DontKnowYetIfMainOrExtension), m_branchesBooked(false) {
    if (desc->className() != "nanoaod::FlatTable")
      throw cms::Exception("Configuration", "NanoAODOutputModule can only write out nanoaod::FlatTable objects");
  }

  void defineBranchesFromFirstEvent(const nanoaod::FlatTable &tab);
  void branch(TTree &tree);

  /// Fill the current table, if extensions == table.extension().
  /// This parameter is used so that the fill is called first for non-extensions and then for extensions
  void fill(const edm::OccurrenceForOutput &iWhatever, TTree &tree, bool extensions);

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
    TBranch *branch;
    NamedBranchPtr(const std::string &aname,
                   const std::string &atitle,
                   const std::string &rootType,
                   TBranch *branchptr = nullptr)
        : name(aname), title(atitle), rootTypeCode(rootType), branch(branchptr) {}
  };
  TBranch *m_counterBranch = nullptr;
  std::vector<NamedBranchPtr> m_int8Branches;
  std::vector<NamedBranchPtr> m_uint8Branches;
  std::vector<NamedBranchPtr> m_int16Branches;
  std::vector<NamedBranchPtr> m_uint16Branches;
  std::vector<NamedBranchPtr> m_int32Branches;
  std::vector<NamedBranchPtr> m_uint32Branches;
  std::vector<NamedBranchPtr> m_floatBranches;
  std::vector<NamedBranchPtr> m_doubleBranches;
  bool m_branchesBooked;

  template <typename T>
  void fillColumn(NamedBranchPtr &pair, const nanoaod::FlatTable &tab) {
    int idx = tab.columnIndex(pair.name);
    if (idx == -1)
      throw cms::Exception("LogicError", "Missing column in input for " + m_baseName + "_" + pair.name);
    pair.branch->SetAddress(
        tab.size() == 0 ? static_cast<T *>(nullptr)
                        : const_cast<T *>(&tab.columnData<T>(idx).front()));  // SetAddress should take a const * !
  }
};

#endif
