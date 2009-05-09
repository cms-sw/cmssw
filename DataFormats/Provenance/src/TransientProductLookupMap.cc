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
// $Id$
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
using namespace edm;

//
// static data member definitions
//

bool CompareTypeInBranchTypeConstBranchDescription::operator()( std::pair<TypeInBranchType,ConstBranchDescription const*> const& iLHS,
                   std::pair<TypeInBranchType,ConstBranchDescription const*> const& iRHS) {
   if(iLHS.first < iRHS.first) {
      return true;
   }
   if(iRHS.first < iLHS.first) {
      return false;
   }
   
   if(iLHS.second->moduleLabel() < iRHS.second->moduleLabel()) {
      return true;
   }
   if(iRHS.second->moduleLabel() < iLHS.second->moduleLabel()) {
      return false;
   }
   
   if(iLHS.second->productInstanceName() < iRHS.second->productInstanceName()) {
      return true;
   }
   if(iRHS.second->productInstanceName() < iLHS.second->productInstanceName()) {
      return false;
   }
   
   return iLHS.second->processName() < iRHS.second->processName();
}

//
// constructors and destructor
//
TransientProductLookupMap::TransientProductLookupMap():
historyIDForBranchType_(static_cast<unsigned int>(NumBranchTypes),ProcessHistoryID()),
processNameOrderingForBranchType_(static_cast<unsigned int>(NumBranchTypes),std::vector<std::string>())
{
}

// TransientProductLookupMap::TransientProductLookupMap(const TransientProductLookupMap& rhs)
// {
//    // do actual copying here;
// }

//TransientProductLookupMap::~TransientProductLookupMap()
//{
//}

//
// assignment operators
//
// const TransientProductLookupMap& TransientProductLookupMap::operator=(const TransientProductLookupMap& rhs)
// {
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
   struct BranchTypeOnlyCompare {
      bool operator()(std::pair<TypeInBranchType,BranchDescriptionIndex> const& iLHS, std::pair<TypeInBranchType,BranchDescriptionIndex> const& iRHS) {
         return iLHS.first.branchType() < iRHS.first.branchType();
      }
   };
   
   struct CompareProcessList {
      std::vector<std::string> const* list_;

      CompareProcessList(std::vector<std::string> const* iList) : list_(iList) {}
      
      bool operator()(ProductLookupIndex const& iLHS, ProductLookupIndex const& iRHS) {
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
fillInProcessIndicies(TransientProductLookupMap::IndexList::iterator iIt, 
                      TransientProductLookupMap::IndexList::iterator iEnd,
                      const std::vector<std::string>& iNameOrder)
{
   //NOTE the iterators are already in the same order as iNameOrder
   std::vector<std::string>::const_reverse_iterator itNO = iNameOrder.rbegin();
   unsigned int index = 0;
   for( ; iIt != iEnd;++iIt) {
      if(iIt->isFirst()) {
         itNO= iNameOrder.rbegin();
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
TransientProductLookupMap::reorderIfNecessary(BranchType iBranch, ProcessHistory const& iHistory, const std::string& iNewProcessName)
{
   ProcessHistoryID& historyID = historyIDForBranchType_[iBranch];
   if(iHistory.id() == historyID) { 
      //std::cout <<"no reordering since history unchanged"<<std::endl;
      return;
   }
   
   if(iHistory.size() == 0) {
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
      while( it != itEnd and itH !=itHEnd ) {
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
      if( !mustReorder && it != itEnd && *it == iNewProcessName) {
         ++it;
      }
      //can only reach the end if we found all the items in the correct order
      if(it == itEnd) { 
         return;
      }
   }

   //must re-sort
   historyID = iHistory.id();
   std::vector<std::string> temp(processNameOrdering.size(),std::string());
   
   
   //we want to add the items at the back
   std::vector<std::string>::reverse_iterator itR = temp.rbegin();
   std::vector<std::string>::reverse_iterator itREnd = temp.rend();
   ProcessHistory::const_reverse_iterator itRH = iHistory.rbegin();
   ProcessHistory::const_reverse_iterator itRHEnd = iHistory.rend();

   if(processNameOrdering.end() != std::find(processNameOrdering.begin(),processNameOrdering.end(),iNewProcessName)) {
      *itR = iNewProcessName;
      ++itR;
   }
   for(; itRH != itRHEnd; ++itRH) {
      if(processNameOrdering.end() != std::find(processNameOrdering.begin(),processNameOrdering.end(),itRH->processName())) {
         
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
      if(temp.end() == std::find(itStart,temp.end(),*itOld)) {
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
   std::equal_range(branchLookup_.begin(),branchLookup_.end(), std::make_pair(TypeInBranchType(TypeID(),iBranch),BranchDescriptionIndex(0)),BranchTypeOnlyCompare());
   
   if(branchRange.first == branchRange.second) {
      return;
   }
   //convert this into Index iterators since that is the structure we must reorder
   IndexList::iterator itIndex = indexList_.begin()+branchRange.first->second;
   IndexList::iterator itIndexEnd = indexList_.end();
   if(branchRange.second != branchLookup_.end()) {
      itIndexEnd = indexList_.begin()+branchRange.second->second;
   }
   
   while(itIndex != itIndexEnd) {
      itIndex->setIsFirst(false);
      IndexList::iterator itNext = itIndex;
      while(itNext != itIndexEnd && not itNext->isFirst()) {
         ++itNext;
      }
      std::sort(itIndex, itNext, CompareProcessList(&processNameOrdering));
      itIndex->setIsFirst(true);
      itIndex = itNext;
   }
   
   //Now that we know all the IDs time to set the values
   fillInProcessIndicies(indexList_.begin()+branchRange.first->second, itIndexEnd, processNameOrdering);

}

void 
TransientProductLookupMap::fillFrom(const FillFromMap& iMap)
{
   assert(processNameOrderingForBranchType_.size()==historyIDForBranchType_.size());

   indexList_.clear();
   indexList_.reserve(iMap.size());
   branchLookup_.clear();
   branchLookup_.reserve(iMap.size()); //this is an upperbound
   
   std::set<std::string, std::greater<std::string> > processNames;
   TypeInBranchType lastSeen(TypeID(),NumBranchTypes);

   //since the actual strings are stored elsewhere, there is no reason to make a copy
   static const std::string kEmpty;
   std::string const* lastSeenModule = &kEmpty;
   std::string const* lastSeenProductInstance = &kEmpty;
   for(FillFromMap::const_iterator it = iMap.begin(), itEnd = iMap.end();
       it != itEnd;
       ++it) {
      bool isFirst =  ( (lastSeen < it->first.first) || (it->first.first < lastSeen) );
      if(isFirst) {
         lastSeen = it->first.first;
         branchLookup_.push_back(std::make_pair(lastSeen,BranchDescriptionIndex(indexList_.size())));
      } else {
         //see if this is the first of a group that only differ by ProcessName
         isFirst = ( *lastSeenModule != it->first.second->moduleLabel() ||
                    *lastSeenProductInstance != it->first.second->productInstanceName() );
      }
      indexList_.push_back( ProductLookupIndex(it->first.second,
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

   std::vector<ProcessHistoryID>::iterator itPH = historyIDForBranchType_.begin();
   std::vector<ProcessHistoryID>::iterator itPHEnd = historyIDForBranchType_.end();
   
   std::vector<std::vector<std::string> >::iterator itPN = processNameOrderingForBranchType_.begin();
   std::vector<std::vector<std::string> >::iterator itPNEnd = processNameOrderingForBranchType_.end();
   
   for(;itPH != itPHEnd; ++itPH, ++itPN) {
      *itPH = ProcessHistoryID();
      itPN->assign(processNames.begin(),processNames.end());
   }

   //Now that we know all the IDs time to set the values
   fillInProcessIndicies(indexList_.begin(), indexList_.end(), processNameOrderingForBranchType_.front());
}

//
// const member functions
//
namespace {
   struct CompareFirst {
      bool operator()(const TransientProductLookupMap::TypeInBranchTypeLookup::value_type& iLHS,
                      const TransientProductLookupMap::TypeInBranchTypeLookup::value_type& iRHS) const {
         return iLHS.first<iRHS.first;
      }
   };
}
std::pair<TransientProductLookupMap::const_iterator, TransientProductLookupMap::const_iterator> 
TransientProductLookupMap::equal_range(const TypeInBranchType& iKey) const
{
   TypeInBranchTypeLookup::const_iterator itFind = std::lower_bound(branchLookup_.begin(),
                                                                      branchLookup_.end(),
                                                                      std::make_pair(iKey,BranchDescriptionIndex(0)),
                                                                      CompareFirst());
   if(itFind == branchLookup_.end() or iKey < itFind->first) {
      return std::make_pair(indexList_.end(),indexList_.end());
   }
   const_iterator itStart = indexList_.begin()+itFind->second;
   const_iterator itEnd = indexList_.end();
   if(++itFind != branchLookup_.end()) {
      itEnd = indexList_.begin()+itFind->second;
   }
   return std::make_pair(itStart,itEnd);
}

//
// static member functions
//
