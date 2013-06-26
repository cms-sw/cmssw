#ifndef DataFormats_Provenance_TypeInBranchType_h
#define DataFormats_Provenance_TypeInBranchType_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     TypeInBranchType
// 
/**\class TypeInBranchType TypeInBranchType.h DataFormats/Provenance/interface/TypeInBranchType.h

 Description: Pairs C++ class type and edm::BranchType

 Usage:
    Used internally to ProductRegistry and for quickly finding data in Principals

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Apr 30 15:50:17 CDT 2009
//

// system include files

// user include files
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/TypeID.h"

// forward declarations

namespace edm {
   class TypeInBranchType {
         
   public:
      TypeInBranchType(TypeID const& iID,
                       BranchType const& iBranch) :
      id_(iID),
      branch_(iBranch) {}
      
      TypeID const& typeID() const {
         return id_;
      }
      
      BranchType const& branchType() const {
         return branch_;
      }
      
      bool operator<(TypeInBranchType const& iRHS) const {
         if(branch_ < iRHS.branch_) {
            return true;
         }
         if(iRHS.branch_ < branch_) {
            return false;
         }
         return id_ < iRHS.id_;
      }
   private:
      TypeID id_;
      BranchType branch_;
   };
   
}

#endif
