#ifndef DataFormats_Provenance_TransientProductLookupMap_h
#define DataFormats_Provenance_TransientProductLookupMap_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     TransientProductLookupMap
// 
/**\class TransientProductLookupMap TransientProductLookupMap.h DataFormats/Provenance/interface/TransientProductLookupMap.h

 Description: Contains information needed to lookup a EDProduct in a Principal

 Usage:
    Filled by ProductRegistry and then used by Principals.
 
 In ProductLookupIndex's held by this class are ordered by
    1) TypeInBranchType [which is BranchType followed by TypeID]
    2) module label
    3) product instance label
    4) process names in descending order of processing [i.e. the latest processing step is the first]

 The 'bool' value 'isFirst' in ProductLookupIndex is set to 'true' if this is the first ProductLookupIndex with the value
 (TypeInBranchType, module label, product instance label).  That is it is the first one in the group where you ignore the 
 process name.
*/
//
// Original Author:  Chris Jones
//         Created:  Fri May  1 11:15:08 CDT 2009
//

// system include files
#include <map>
#include <vector>

// user include files
#include "DataFormats/Provenance/interface/TypeInBranchType.h"
#include "DataFormats/Provenance/interface/BranchDescriptionIndex.h"
#include "DataFormats/Provenance/interface/ProductLookupIndex.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

// forward declarations
namespace edm {
   class ProcessHistory;
   
   struct CompareTypeInBranchTypeConstBranchDescription {
      bool operator()(std::pair<TypeInBranchType, ConstBranchDescription const*> const& iLHS,
                      std::pair<TypeInBranchType, ConstBranchDescription const*> const& iRHS) const;
   };
   
   class TransientProductLookupMap {
      
   public:
      typedef std::vector<std::pair<TypeInBranchType, BranchDescriptionIndex> > TypeInBranchTypeLookup;
      typedef std::vector<ProductLookupIndex> ProductLookupIndexList;
      
      typedef ProductLookupIndexList::const_iterator const_iterator;

      typedef std::map<std::pair<TypeInBranchType, ConstBranchDescription const*>,
			ProductTransientIndex,
			CompareTypeInBranchTypeConstBranchDescription> FillFromMap;

      TransientProductLookupMap();
            
      // ---------- const member functions ---------------------
      
      ///returns a pair of iterators that define the range for items matching the TypeInBranchType
      std::pair<const_iterator, const_iterator> equal_range(TypeInBranchType const&) const;
      
      ///returns a pair of iterators that define the range for items matching
      ///the TypeInBranchType, the module label, and the product instance name
      std::pair<const_iterator, const_iterator> equal_range(TypeInBranchType const&,
							    std::string const&,
							    std::string const&) const;
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------

      ///reorders the ProductLookupIndexes for the BranchType based on the processing ordering
      void reorderIfNecessary(BranchType, ProcessHistory const&, std::string const& iNewProcessName);
      
      void fillFrom(FillFromMap const&);

      const_iterator begin() const {return productLookupIndexList_.begin();}

      const_iterator end() const {return productLookupIndexList_.end();}

      int fillCount() const {return fillCount_;}

   private:
      // ---------- member data --------------------------------
      TypeInBranchTypeLookup branchLookup_;
      ProductLookupIndexList productLookupIndexList_;
      std::vector<ProcessHistoryID> historyIDsForBranchType_;
      std::vector<std::vector<std::string> > processNameOrderingForBranchType_;
      int fillCount_;
   };
   
}

#endif
