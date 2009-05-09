#ifndef DataFormats_Provenance_ProductLookupIndex_h
#define DataFormats_Provenance_ProductLookupIndex_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     ProductLookupIndex
// 
/**\class ProductLookupIndex ProductLookupIndex.h DataFormats/Provenance/interface/ProductLookupIndex.h

 Description: Contains information needed to lookup a EDProduct in a Principal

 Usage:
    Used internally by ProductRegistry and Principals.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri May  1 11:00:21 CDT 2009
// $Id$
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/ProductTransientIndex.h"

// forward declarations

namespace  edm {
   class ConstBranchDescription;
   
   class ProductLookupIndex {
         
   public:
      ProductLookupIndex(ConstBranchDescription const* iBranch,
                         ProductTransientIndex iIndex,
                         unsigned int iProcessIndex,
                         bool iIsFirst=false):
      branchDescription_(iBranch),
      index_(iIndex),
      processIndex_(iProcessIndex),
      isFirst_(iIsFirst) {}
      //virtual ~ProductLookupIndex();
      
      void setIsFirst(bool iIsFirst) {
         isFirst_ = iIsFirst;
      }

      void setProcessIndex(unsigned int iIndex) {
         processIndex_ = iIndex;
      }
      // ---------- const member functions ---------------------
      ConstBranchDescription const* branchDescription() const { return branchDescription_; }
      ProductTransientIndex index() const { return index_;}
      
      ///index into the process history table for the process corresponding to this element
      unsigned int processIndex() const {return processIndex_;}
      
      /**True if this is the first ProductLookupIndex in the container for a series of elements
       who are only different based on the ProcessName
       */
      bool isFirst() const { return isFirst_;}
      
   private:
      //ProductLookupIndex(const ProductLookupIndex&); // allow default
      
      //const ProductLookupIndex& operator=(const ProductLookupIndex&); // allow default
      
      // ---------- member data --------------------------------
      ConstBranchDescription const* branchDescription_;
      ProductTransientIndex index_;
      unsigned int processIndex_;
      bool isFirst_;
      
   };   
}


#endif
