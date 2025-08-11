#include "L1TriggerScouting/Utilities/plugins/SelectedBxTableOutputBranches.h"

void SelectedBxTableOutputBranches::beginFill(const edm::OccurrenceForOutput &iWhatever, TTree &tree) {
  if (m_branch == nullptr) {
    m_branch = tree.Branch(m_name.c_str(), &m_value, (m_name + "/O").c_str());
    m_branch->SetTitle("SelectionBx bit");
  }
  iWhatever.getByToken(m_token, m_handle);
  m_bitset.reset();
  for (unsigned bx : *m_handle) {
    m_bitset[bx] = true;
  }
}
