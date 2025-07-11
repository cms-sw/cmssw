#ifndef L1TriggerScouting_Utilities_SelectedBxTableOutputBranches_h
#define L1TriggerScouting_Utilities_SelectedBxTableOutputBranches_h

#include <string>
#include <vector>
#include <bitset>
#include <TTree.h>
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "DataFormats/L1Scouting/interface/OrbitFlatTable.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class SelectedBxTableOutputBranches {
public:
  SelectedBxTableOutputBranches(const edm::BranchDescription *desc, const edm::EDGetToken &token)
      : m_token(token), m_name("SelBx_" + desc->moduleLabel()), m_branch(nullptr) {
    if (desc->className() != "std::vector<unsigned int>")
      throw cms::Exception("Configuration", "SelectedBxTableOutputBranches can only write out vector<unsigned int>");
    if (desc->productInstanceName() != "SelBx") {
      m_name += "_" + desc->productInstanceName();
    }
  }

  void beginFill(const edm::OccurrenceForOutput &iWhatever, TTree &tree);
  void fillBx(uint32_t bx) { m_value = m_bitset[bx]; }
  void endFill() {}

private:
  edm::EDGetToken m_token;
  std::string m_name;
  bool m_value;
  std::bitset<l1ScoutingRun3::OrbitFlatTable::NBX> m_bitset;
  TBranch *m_branch;

  edm::Handle<std::vector<unsigned int>> m_handle;
};

#endif
