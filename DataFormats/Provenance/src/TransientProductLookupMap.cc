// -*- C++ -*-
//
// Package:     Provenance
// Class  :     TransientProductLookupMap
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri May  1 12:17:12 CDT 2009
//

// system include files
#include <algorithm>

// user include files
#include "DataFormats/Provenance/interface/TransientProductLookupMap.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

namespace edm {
  bool CompareTypeInBranchTypeConstBranchDescription::operator()(std::pair<TypeInBranchType, ConstBranchDescription const*> const& iLHS,
                     std::pair<TypeInBranchType, ConstBranchDescription const*> const& iRHS) const {
     if(iLHS.first < iRHS.first) {
        return true;
     }
     if(iRHS.first < iLHS.first) {
        return false;
     }
     
     int c = iLHS.second->moduleLabel().compare(iRHS.second->moduleLabel());
     if(c < 0) {
        return true;
     }
     if(c > 0) {
        return false;
     }
     c = iLHS.second->productInstanceName().compare(iRHS.second->productInstanceName());
     if(c < 0) {
        return true;
     }
     if(c > 0) {
        return false;
     }
     
     return iLHS.second->processName() < iRHS.second->processName();
  }
  
  //
  // constructors and destructor
  //
  TransientProductLookupMap::TransientProductLookupMap() :
      historyIDsForBranchType_(static_cast<unsigned int>(NumBranchTypes), ProcessHistoryID()),
      processNameOrderingForBranchType_(static_cast<unsigned int>(NumBranchTypes), std::vector<std::string>()) {
  }
  
  // TransientProductLookupMap::TransientProductLookupMap(TransientProductLookupMap const& rhs) {
  //    // do actual copying here;
  // }
  
  //TransientProductLookupMap::~TransientProductLookupMap() {
  //}
  
  //
  // assignment operators
  //
  // TransientProductLookupMap const& TransientProductLookupMap::operator=(TransientProductLookupMap const& rhs) {
  //   //An exception safe implementation is
  //   TransientProductLookupMap temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }
  
  //
  // member functions
  //
  namespace  {

     struct CompareModuleLabelAndProductInstanceName {
        typedef std::pair<std::string const*, std::string const*> StringPtrPair;
        bool operator()(StringPtrPair const& iLHS, StringPtrPair const& iRHS) const {
           int c = iLHS.first->compare(*iRHS.first);
           if (c < 0) return true;
           if (c > 0) return false;
           return(*iLHS.second < *iRHS.second);
        }
        bool operator()(ProductLookupIndex const& iLHS, StringPtrPair const& iRHS) const {
           int c = iLHS.branchDescription()->moduleLabel().compare(*iRHS.first);
           if (c < 0) return true;
           if (c > 0) return false;
           return(iLHS.branchDescription()->productInstanceName() < *iRHS.second);
        }
        bool operator()(StringPtrPair const& iLHS, ProductLookupIndex const& iRHS) const {
           int c = iLHS.first->compare(iRHS.branchDescription()->moduleLabel());
           if (c < 0) return true;
           if (c > 0) return false;
           return(*iLHS.second < iRHS.branchDescription()->productInstanceName());
        }
        bool operator()(ProductLookupIndex const& iLHS, ProductLookupIndex const& iRHS) const {
           int c = iLHS.branchDescription()->moduleLabel().compare(iRHS.branchDescription()->moduleLabel());
           if (c < 0) return true;
           if (c > 0) return false;
           return(iLHS.branchDescription()->productInstanceName() < iRHS.branchDescription()->productInstanceName());
        }
     };

     struct BranchTypeOnlyCompare {
        bool operator()(std::pair<TypeInBranchType, BranchDescriptionIndex> const& iLHS, std::pair<TypeInBranchType, BranchDescriptionIndex> const& iRHS) const {
           return iLHS.first.branchType() < iRHS.first.branchType();
        }
     };
     
     struct CompareProcessList {
        std::vector<std::string> const* list_;
  
        CompareProcessList(std::vector<std::string> const* iList) : list_(iList) {}
        
        bool operator()(ProductLookupIndex const& iLHS, ProductLookupIndex const& iRHS) const {
           std::string const& lhs = iLHS.branchDescription()->processName();
           std::string const& rhs = iRHS.branchDescription()->processName();
           if(lhs == rhs) {return false;}
           //NOTE: names in the vector are oldest to newest and we want to order by newest
           for(std::vector<std::string>::const_reverse_iterator it = list_->rbegin(), itEnd =list_->rend();
               it != itEnd;
               ++it){
              if(*it == lhs) {return true;}
              if(*it == rhs) { return false;}
           }
           return false;
        }
     };
  }
  
  static
  void
  fillInProcessIndexes(TransientProductLookupMap::ProductLookupIndexList::iterator iIt, 
                        TransientProductLookupMap::ProductLookupIndexList::iterator iEnd,
                        std::vector<std::string> const& iNameOrder) {
     //NOTE the iterators are already in the same order as iNameOrder
     std::vector<std::string>::const_reverse_iterator itNO = iNameOrder.rbegin();
     unsigned int index = 0;
     for(; iIt != iEnd; ++iIt) {
        if(iIt->isFirst()) {
           itNO = iNameOrder.rbegin();
           index = 0;
        }
        while(*itNO != iIt->branchDescription()->processName()) {
           ++itNO;
           assert(itNO != iNameOrder.rend());
           ++index;
        }
        iIt->setProcessIndex(index);
     }
  }
  
  void 
  TransientProductLookupMap::reorderIfNecessary(BranchType iBranch, ProcessHistory const& iHistory, std::string const& iNewProcessName) {

     ProcessHistoryID& historyID = historyIDsForBranchType_[iBranch];
     if(iHistory.id() == historyID) { 
        //std::cout <<"no reordering since history unchanged"<<std::endl;
        return;
     }
     
     if(iHistory.empty()) {
        //std::cout <<"no reordering since history empty"<<std::endl;
        historyID = iHistory.id(); 
        return; 
     }
     std::vector<std::string>& processNameOrdering = processNameOrderingForBranchType_[iBranch];
  
     //iHistory may be missing entries in processNameOrdering if two files were merged together and one file
     // had fewer processing steps than the other one.
     //iHistory may have more entries than processNameOrdering if all data products for those extra entries
     // were dropped
     
     //if iHistory already in same order as processNameOrdering than we don't have to do anything
     std::vector<std::string>::iterator it = processNameOrdering.begin();
     std::vector<std::string>::iterator itEnd = processNameOrdering.end();
     ProcessHistory::const_iterator itH = iHistory.begin();
     ProcessHistory::const_iterator itHEnd = iHistory.end();
     
     {
        std::vector<std::string>::iterator itStart = it;
        bool mustReorder = false;
        while(it != itEnd && itH != itHEnd) {
           if(*it == itH->processName()) {
              ++it;
           } else {
              //see if we already passed it
              for(std::vector<std::string>::iterator itOld = itStart; itOld != it; ++itOld) {
                 if(*itOld == itH->processName()) {
                    mustReorder = true;
                    break;
                 }
              }
              if(mustReorder) {
                 break;
              }
           }
           ++itH;
        }
        if(!mustReorder && it != itEnd && *it == iNewProcessName) {
           ++it;
        }
        //can only reach the end if we found all the items in the correct order
        if(it == itEnd) { 
           return;
        }
     }
  
     //must re-sort
     historyID = iHistory.id();
     std::vector<std::string> temp(processNameOrdering.size(), std::string());
     
     
     //we want to add the items at the back
     std::vector<std::string>::reverse_iterator itR = temp.rbegin();
     std::vector<std::string>::reverse_iterator itREnd = temp.rend();
     ProcessHistory::const_reverse_iterator itRH = iHistory.rbegin();
     ProcessHistory::const_reverse_iterator itRHEnd = iHistory.rend();
  
     if(processNameOrdering.end() != std::find(processNameOrdering.begin(), processNameOrdering.end(), iNewProcessName)) {
        *itR = iNewProcessName;
        ++itR;
     }
     for(; itRH != itRHEnd; ++itRH) {
        if(processNameOrdering.end() != std::find(processNameOrdering.begin(), processNameOrdering.end(), itRH->processName())) {
           
           *itR = itRH->processName();
           ++itR;
        }
     }
  
     //have to fill in the missing processes from processNameOrdering_
     // we do this at the beginning because we lookup data in the reverse order
     // so we want the ones we know are there to be searched first
     std::vector<std::string>::iterator itOld = processNameOrdering.begin();
     it = temp.begin();
     std::vector<std::string>::iterator itStart = temp.begin()+(itREnd - itR);
     while(it != itStart) {
        assert(itOld != processNameOrdering.end());
        if(temp.end() == std::find(itStart, temp.end(), *itOld)) {
           //didn't find it so need to add this to our list
           *it = *itOld;
           ++it;
        }
        ++itOld;
     }
     
     processNameOrdering.swap(temp);
     
     //now we need to go through our data structure and change the processing orders
     //first find the range for this BranchType
     std::pair<TypeInBranchTypeLookup::iterator, TypeInBranchTypeLookup::iterator> branchRange = 
     std::equal_range(branchLookup_.begin(), branchLookup_.end(), std::make_pair(TypeInBranchType(TypeID(), iBranch), BranchDescriptionIndex(0)), BranchTypeOnlyCompare());
     
     if(branchRange.first == branchRange.second) {
        return;
     }
     //convert this into Index iterators since that is the structure we must reorder
     ProductLookupIndexList::iterator itIndex = productLookupIndexList_.begin()+branchRange.first->second;
     ProductLookupIndexList::iterator itIndexEnd = productLookupIndexList_.end();
     if(branchRange.second != branchLookup_.end()) {
        itIndexEnd = productLookupIndexList_.begin()+branchRange.second->second;
     }
     
     while(itIndex != itIndexEnd) {
        itIndex->setIsFirst(false);
        ProductLookupIndexList::iterator itNext = itIndex;
        while(itNext != itIndexEnd && !itNext->isFirst()) {
           ++itNext;
        }
        std::sort(itIndex, itNext, CompareProcessList(&processNameOrdering));
        itIndex->setIsFirst(true);
        itIndex = itNext;
     }
     
     //Now that we know all the IDs time to set the values
     fillInProcessIndexes(productLookupIndexList_.begin()+branchRange.first->second, itIndexEnd, processNameOrdering);
  
  }
  
  void 
  TransientProductLookupMap::fillFrom(FillFromMap const& iMap) {

     assert(processNameOrderingForBranchType_.size()==historyIDsForBranchType_.size());
  
     productLookupIndexList_.clear();
     productLookupIndexList_.reserve(iMap.size());
     branchLookup_.clear();
     branchLookup_.reserve(iMap.size()); //this is an upperbound
     
     std::set<std::string, std::greater<std::string> > processNames;
     TypeInBranchType lastSeen(TypeID(), NumBranchTypes);
  
     //since the actual strings are stored elsewhere, there is no reason to make a copy
     static std::string const kEmpty;
     std::string const* lastSeenModule = &kEmpty;
     std::string const* lastSeenProductInstance = &kEmpty;
     for(FillFromMap::const_iterator it = iMap.begin(), itEnd = iMap.end();
         it != itEnd;
         ++it) {
        bool isFirst =  ((lastSeen < it->first.first) || (it->first.first < lastSeen));
        if(isFirst) {
           lastSeen = it->first.first;
           branchLookup_.push_back(std::make_pair(lastSeen, BranchDescriptionIndex(productLookupIndexList_.size())));
        } else {
           //see if this is the first of a group that only differ by ProcessName
           isFirst = (*lastSeenModule != it->first.second->moduleLabel() ||
                      *lastSeenProductInstance != it->first.second->productInstanceName());
        }
        productLookupIndexList_.push_back(ProductLookupIndex(it->first.second,
                                                it->second,
                                                0,
                                                isFirst)
                            );
        if(isFirst) {
           lastSeenModule = &(it->first.second->moduleLabel());
           lastSeenProductInstance = &(it->first.second->productInstanceName());         
        }
        processNames.insert(it->first.second->processName());
     }
  
     std::vector<ProcessHistoryID>::iterator itPH = historyIDsForBranchType_.begin();
     std::vector<ProcessHistoryID>::iterator itPHEnd = historyIDsForBranchType_.end();
     
     std::vector<std::vector<std::string> >::iterator itPN = processNameOrderingForBranchType_.begin();
     std::vector<std::vector<std::string> >::iterator itPNEnd = processNameOrderingForBranchType_.end();
     
     for(;itPH != itPHEnd; ++itPH, ++itPN) {
        *itPH = ProcessHistoryID();
        itPN->assign(processNames.begin(), processNames.end());
     }
  
     //Now that we know all the IDs time to set the values
     fillInProcessIndexes(productLookupIndexList_.begin(), productLookupIndexList_.end(), processNameOrderingForBranchType_.front());
  }
  
  //
  // const member functions
  //
  namespace {
     struct CompareFirst {
        bool operator()(TransientProductLookupMap::TypeInBranchTypeLookup::value_type const& iLHS,
                        TransientProductLookupMap::TypeInBranchTypeLookup::value_type const& iRHS) const {
           return iLHS.first < iRHS.first;
        }
     };
  }

  std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> 
  TransientProductLookupMap::equal_range(TypeInBranchType const& iKey) const {
     TypeInBranchTypeLookup::const_iterator itFind = std::lower_bound(branchLookup_.begin(),
                                                                      branchLookup_.end(),
                                                                      std::make_pair(iKey, BranchDescriptionIndex(0)),
                                                                      CompareFirst());
     if(itFind == branchLookup_.end() || iKey < itFind->first) {
        return std::make_pair(productLookupIndexList_.end(), productLookupIndexList_.end());
     }
     const_iterator itStart = productLookupIndexList_.begin() + itFind->second;
     const_iterator itEnd = productLookupIndexList_.end();
     if(++itFind != branchLookup_.end()) {
        itEnd = productLookupIndexList_.begin() + itFind->second;
     }
     return std::make_pair(itStart, itEnd);
  }
  
  std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> 
  TransientProductLookupMap::equal_range(TypeInBranchType const& iKey,
	 std::string const& moduleLabel,
	 std::string const& productInstanceName) const {
     std::pair<const_iterator, const_iterator> itPair = this->equal_range(iKey);

     if (itPair.first == itPair.second) {
        return itPair;
     }

     // Advance lower bound only
     itPair.first = std::lower_bound(itPair.first, itPair.second, std::make_pair(&moduleLabel, &productInstanceName), CompareModuleLabelAndProductInstanceName());  
     // Protect against no match
     if (!(itPair.first < itPair.second) ||
         itPair.first->branchDescription()->moduleLabel() != moduleLabel ||
         itPair.first->branchDescription()->productInstanceName() != productInstanceName) {
       itPair.second = itPair.first;
     }
     return itPair;
  }
  
  //
  // static member functions
  //
}
