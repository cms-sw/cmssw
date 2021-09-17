#ifndef FWCore_Framework_EarlyDeleteHelper_h
#define FWCore_Framework_EarlyDeleteHelper_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EarlyDeleteHelper
//
/**\class EarlyDeleteHelper EarlyDeleteHelper.h FWCore/Framework/interface/EarlyDeleteHelper.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  6 16:17:08 CST 2012
//

// system include files
#include <vector>
#include <atomic>

// user include files
#include "DataFormats/Provenance/interface/BranchID.h"
// forward declarations
namespace edm {
  class EventPrincipal;

  struct BranchToCount {
    edm::BranchID const branch;
    std::atomic<unsigned int> count;

    BranchToCount(edm::BranchID id, unsigned int count) : branch(id), count(count) {}

    BranchToCount(BranchToCount const& iOther) : branch(iOther.branch), count(iOther.count.load()) {}
  };

  class EarlyDeleteHelper {
  public:
    EarlyDeleteHelper(unsigned int* iBeginIndexItr,
                      unsigned int* iEndIndexItr,
                      std::vector<BranchToCount>* iBranchCounts);
    EarlyDeleteHelper(const EarlyDeleteHelper&);
    EarlyDeleteHelper& operator=(const EarlyDeleteHelper&) = delete;
    //virtual ~EarlyDeleteHelper();

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void reset() { pathsLeftToComplete_ = nPathsOn_; }
    void moduleRan(EventPrincipal const&);
    void pathFinished(EventPrincipal const&);
    void addedToPath() { ++nPathsOn_; }
    void appendIndex(unsigned int index);
    void shiftIndexPointers(unsigned int iShift);

    unsigned int* begin() { return pBeginIndex_; }
    unsigned int* end() { return pEndIndex_; }

  private:
    // ---------- member data --------------------------------
    unsigned int* pBeginIndex_;
    unsigned int* pEndIndex_;
    std::vector<BranchToCount>* pBranchCounts_;
    std::atomic<unsigned int> pathsLeftToComplete_;
    unsigned int nPathsOn_;
  };
}  // namespace edm

#endif
