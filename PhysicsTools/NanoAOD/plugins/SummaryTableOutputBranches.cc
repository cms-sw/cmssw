#include "PhysicsTools/NanoAOD/plugins/SummaryTableOutputBranches.h"

template <typename T, typename Col>
void SummaryTableOutputBranches::makeScalarBranches(const std::vector<Col> &tabcols,
                                                    TTree &tree,
                                                    const std::string &rootType,
                                                    std::vector<NamedBranchPtr> &branches) {
  for (const auto &col : tabcols) {
    if (std::find_if(branches.begin(), branches.end(), [&col](const NamedBranchPtr &x) {
          return x.name == col.name;
        }) == branches.end()) {
      T backFillValue = 0;
      auto *br = tree.Branch(col.name.c_str(), &backFillValue, (col.name + "/" + rootType).c_str());
      br->SetTitle(col.doc.c_str());
      for (unsigned long i = 0; i < m_fills; i++)
        br->Fill();
      branches.emplace_back(col.name, br);
    }
  }
}

template <typename Col>
void SummaryTableOutputBranches::makeVectorBranches(const std::vector<Col> &tabcols,
                                                    TTree &tree,
                                                    const std::string &rootType,
                                                    std::vector<NamedVectorBranchPtr> &branches) {
  for (const auto &col : tabcols) {
    if (std::find_if(branches.begin(), branches.end(), [&col](const NamedBranchPtr &x) {
          return x.name == col.name;
        }) == branches.end()) {
      int backFillValue = 0;
      auto *cbr = tree.Branch(("n" + col.name).c_str(), &backFillValue, ("n" + col.name + "/I").c_str());
      auto *vbr =
          tree.Branch(col.name.c_str(), (void *)nullptr, (col.name + "[n" + col.name + "]/" + rootType).c_str());
      cbr->SetTitle(("Number of entries in " + col.name).c_str());
      vbr->SetTitle(col.doc.c_str());
      for (unsigned long i = 0; i < m_fills; i++) {
        cbr->Fill();
        vbr->Fill();
      }
      branches.emplace_back(col.name, cbr, vbr);
    }
  }
}

template <typename Col>
void SummaryTableOutputBranches::fillScalarBranches(const std::vector<Col> &tabcols,
                                                    std::vector<NamedBranchPtr> &branches) {
  if (tabcols.size() != branches.size())
    throw cms::Exception("LogicError", "Mismatch in table columns");
  for (unsigned int i = 0, n = tabcols.size(); i < n; ++i) {
    if (tabcols[i].name != branches[i].name)
      throw cms::Exception("LogicError", "Mismatch in table columns");
    branches[i].branch->SetAddress(const_cast<typename Col::value_type *>(&tabcols[i].value));
  }
}

template <typename Col>
void SummaryTableOutputBranches::fillVectorBranches(const std::vector<Col> &tabcols,
                                                    std::vector<NamedVectorBranchPtr> &branches) {
  if (tabcols.size() != branches.size())
    throw cms::Exception("LogicError", "Mismatch in table columns");
  for (unsigned int i = 0, n = tabcols.size(); i < n; ++i) {
    if (tabcols[i].name != branches[i].name)
      throw cms::Exception("LogicError", "Mismatch in table columns");
    branches[i].count = tabcols[i].values.size();
    branches[i].branch->SetAddress(const_cast<typename Col::element_type *>(&tabcols[i].values.front()));
  }
}

void SummaryTableOutputBranches::updateBranches(const nanoaod::MergeableCounterTable &tab, TTree &tree) {
  makeScalarBranches<Long64_t>(tab.intCols(), tree, "L", m_intBranches);
  makeScalarBranches<Double_t>(tab.floatCols(), tree, "D", m_floatBranches);
  makeScalarBranches<Double_t>(tab.floatWithNormCols(), tree, "D", m_floatWithNormBranches);
  makeVectorBranches(tab.vintCols(), tree, "L", m_vintBranches);
  makeVectorBranches(tab.vfloatCols(), tree, "D", m_vfloatBranches);
  makeVectorBranches(tab.vfloatWithNormCols(), tree, "D", m_vfloatWithNormBranches);

  // now we go set the pointers for the counter branches
  for (auto &vbp : m_vintBranches)
    vbp.counterBranch->SetAddress(&vbp.count);
  for (auto &vbp : m_vfloatBranches)
    vbp.counterBranch->SetAddress(&vbp.count);
  for (auto &vbp : m_vfloatWithNormBranches)
    vbp.counterBranch->SetAddress(&vbp.count);
}

void SummaryTableOutputBranches::fill(const edm::OccurrenceForOutput &iWhatever, TTree &tree) {
  edm::Handle<nanoaod::MergeableCounterTable> handle;
  iWhatever.getByToken(m_token, handle);
  const nanoaod::MergeableCounterTable &tab = *handle;

  updateBranches(tab, tree);

  fillScalarBranches(tab.intCols(), m_intBranches);
  fillScalarBranches(tab.floatCols(), m_floatBranches);
  fillScalarBranches(tab.floatWithNormCols(), m_floatWithNormBranches);
  fillVectorBranches(tab.vintCols(), m_vintBranches);
  fillVectorBranches(tab.vfloatCols(), m_vfloatBranches);
  fillVectorBranches(tab.vfloatWithNormCols(), m_vfloatWithNormBranches);
  m_fills++;
}
