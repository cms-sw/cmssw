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

// user include files

// forward declarations
namespace edm {
  class BranchID;
  class EventPrincipal;
  
  class EarlyDeleteHelper
  {
    
  public:
    EarlyDeleteHelper(unsigned int* iBeginIndexItr,
                      unsigned int* iEndIndexItr,
                      std::vector<std::pair<edm::BranchID,unsigned int>>* iBranchCounts);
    EarlyDeleteHelper(const EarlyDeleteHelper&) = default;
        EarlyDeleteHelper& operator=(const EarlyDeleteHelper&) = default;
    //virtual ~EarlyDeleteHelper();
    
    // ---------- const member functions ---------------------
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    void reset() {pathsLeftToComplete_ = nPathsOn_;}
    void moduleRan(EventPrincipal&);
    void pathFinished(EventPrincipal&);
    void addedToPath() { ++nPathsOn_;}
    void appendIndex(unsigned int index);
    void shiftIndexPointers(unsigned int iShift);
    
    unsigned int* begin() { return pBeginIndex_;}
    unsigned int* end() { return pEndIndex_;}
    
  private:
    
    // ---------- member data --------------------------------
    unsigned int* pBeginIndex_;
    unsigned int* pEndIndex_;
    std::vector<std::pair<edm::BranchID,unsigned int>>* pBranchCounts_;
    unsigned int pathsLeftToComplete_;
    unsigned int nPathsOn_;
    
  };
}


#endif
