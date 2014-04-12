// -*- C++ -*-
//
// Package:     Framework
// Class  :     EarlyDeleteHelper
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  6 16:17:10 CST 2012
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "FWCore/Framework/src/EarlyDeleteHelper.h"

using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EarlyDeleteHelper::EarlyDeleteHelper(unsigned int* iBeginIndexItr,
                                     unsigned int* iEndIndexItr,
                                     std::vector<std::pair<edm::BranchID,unsigned int>>* iBranchCounts):
pBeginIndex_(iBeginIndexItr),
pEndIndex_(iEndIndexItr),
pBranchCounts_(iBranchCounts),
pathsLeftToComplete_(0),
nPathsOn_(0)
{
}

// EarlyDeleteHelper::EarlyDeleteHelper(const EarlyDeleteHelper& rhs)
// {
//    // do actual copying here;
// }

//EarlyDeleteHelper::~EarlyDeleteHelper()
//{
//}

//
// assignment operators
//
// const EarlyDeleteHelper& EarlyDeleteHelper::operator=(const EarlyDeleteHelper& rhs)
// {
//   //An exception safe implementation is
//   EarlyDeleteHelper temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
EarlyDeleteHelper::moduleRan(EventPrincipal& iEvent) {
  pathsLeftToComplete_=0;
  for(auto it = pBeginIndex_; it != pEndIndex_;++it) {
    auto& count = (*pBranchCounts_)[*it];
    assert(count.second>0);
    --(count.second);
    if(count.second==0) {
      iEvent.deleteProduct(count.first);
    }
  }
}

void 
EarlyDeleteHelper::pathFinished(EventPrincipal& iEvent) {
  if(pathsLeftToComplete_>0 && --pathsLeftToComplete_ == 0) {
    //we can never reach this module now so declare it as run
    moduleRan(iEvent);
  }
}

void
EarlyDeleteHelper::appendIndex(unsigned int iIndex) {
  *pEndIndex_=iIndex;
  ++pEndIndex_;
}

void
EarlyDeleteHelper::shiftIndexPointers(unsigned int iShift) {
  pEndIndex_ -=iShift;
  pBeginIndex_ -=iShift;
}

//
// const member functions
//

//
// static member functions
//
