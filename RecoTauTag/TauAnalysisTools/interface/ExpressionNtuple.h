/*
 * Tool to build a TTree of columns of from StringObjectFunctions.
 *
 * Author: Evan K. Friis, UW Madison
 *
 */

#ifndef EXPRESSIONNTUPLE_E32UGXK7
#define EXPRESSIONNTUPLE_E32UGXK7

#include "boost/utility.hpp"
#include <boost/ptr_container/ptr_vector.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "TTree.h"

#include "RecoTauTag/TauAnalysisTools/interface/ExpressionNtupleColumn.h"

// so we don't conflict with FinalStateAnalysis
namespace uct {

template<class T>
class ExpressionNtuple : private boost::noncopyable {
  public:
    ExpressionNtuple(const edm::ParameterSet& pset);
    ~ExpressionNtuple();

    // Setup the tree in the given TFile
    void initialize(TFileDirectory& fs);
    // Fill the tree with an element with given index.
    void fill(const T& element, int idx = -1);
    // Get access to the internal tree
    TTree* tree() const { return tree_; }
  private:
    TTree* tree_;
    std::vector<std::string> columnNames_;
    edm::ParameterSet pset_;
    boost::ptr_vector<ExpressionNtupleColumn<T> > columns_;
    boost::shared_ptr<Int_t> idxBranch_;
};

template<class T>
ExpressionNtuple<T>::ExpressionNtuple(const edm::ParameterSet& pset):
  pset_(pset) {
  tree_ = NULL;
  typedef std::vector<std::string> vstring;
  // Double check no column already exists
  columnNames_ = pset.getParameterNames();
  std::set<std::string> enteredAlready;
  for (size_t i = 0; i < columnNames_.size(); ++i) {
    const std::string& colName = columnNames_[i];
    if (enteredAlready.count(colName)) {
      throw cms::Exception("DuplicatedBranch")
        << " The ntuple branch with name " << colName
        << " has already been registered!" << std::endl;
    }
    enteredAlready.insert(colName);
  }
  idxBranch_.reset(new Int_t);
}

template<class T> ExpressionNtuple<T>::~ExpressionNtuple() {}

template<class T> void ExpressionNtuple<T>::initialize(TFileDirectory& fs) {
  tree_ = fs.make<TTree>("Ntuple", "Expression Ntuple");
  // Build branches
  for (size_t i = 0; i < columnNames_.size(); ++i) {
    columns_.push_back(buildColumn<T>(columnNames_[i], pset_, tree_));
  }
  // A special branch so we know which subrow we are on.
  tree_->Branch("idx", idxBranch_.get(), "idx/I");
}

template<class T> void ExpressionNtuple<T>::fill(const T& element, int idx) {
  for (size_t i = 0; i < columns_.size(); ++i) {
    // Compute the function and load the value into the column.
    columns_[i].compute(element);
  }
  *idxBranch_ = idx;
  tree_->Fill();
}

}

#endif /* end of include guard: EXPRESSIONNTUPLE_E32UGXK7 */

