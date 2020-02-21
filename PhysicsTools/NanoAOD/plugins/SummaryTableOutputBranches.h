#ifndef PhysicsTools_NanoAOD_SummaryTableOutputBranches_h
#define PhysicsTools_NanoAOD_SummaryTableOutputBranches_h

#include <string>
#include <vector>
#include <TTree.h>
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class SummaryTableOutputBranches {
public:
  SummaryTableOutputBranches(const edm::BranchDescription *desc, const edm::EDGetToken &token)
      : m_token(token), m_fills(0) {
    if (desc->className() != "nanoaod::MergeableCounterTable")
      throw cms::Exception("Configuration", "NanoAODOutputModule can only write out MergableCounterTable objects");
  }

  void fill(const edm::OccurrenceForOutput &iWhatever, TTree &tree);

private:
  edm::EDGetToken m_token;

  struct NamedBranchPtr {
    std::string name;
    TBranch *branch;
    NamedBranchPtr(const std::string &aname, TBranch *branchptr = nullptr) : name(aname), branch(branchptr) {}
  };
  std::vector<NamedBranchPtr> m_intBranches, m_floatBranches, m_floatWithNormBranches;

  struct NamedVectorBranchPtr : public NamedBranchPtr {
    UInt_t count;
    TBranch *counterBranch;
    NamedVectorBranchPtr(const std::string &aname,
                         TBranch *counterBranchptr = nullptr,
                         TBranch *valueBranchptr = nullptr)
        : NamedBranchPtr(aname, valueBranchptr), counterBranch(counterBranchptr) {}
  };
  std::vector<NamedVectorBranchPtr> m_vintBranches, m_vfloatBranches, m_vfloatWithNormBranches;

  unsigned long m_fills;

  void updateBranches(const nanoaod::MergeableCounterTable &tab, TTree &tree);

  template <typename T, typename Col>
  void makeScalarBranches(const std::vector<Col> &tabcols,
                          TTree &tree,
                          const std::string &rootType,
                          std::vector<NamedBranchPtr> &branches);
  template <typename Col>
  void makeVectorBranches(const std::vector<Col> &tabcols,
                          TTree &tree,
                          const std::string &rootType,
                          std::vector<NamedVectorBranchPtr> &branches);

  template <typename Col>
  void fillScalarBranches(const std::vector<Col> &tabcols, std::vector<NamedBranchPtr> &branches);
  template <typename Col>
  void fillVectorBranches(const std::vector<Col> &tabcols, std::vector<NamedVectorBranchPtr> &branches);
};

#endif
