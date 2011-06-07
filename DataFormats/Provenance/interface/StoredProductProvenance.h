#ifndef DataFormats_Provenance_StoredProductProvenance_h
#define DataFormats_Provenance_StoredProductProvenance_h
// -*- C++ -*-
//
// Package:     Provenance
// Class  :     StoredProductProvenance
//
/**\class StoredProductProvenance StoredProductProvenance.h DataFormats/Provenance/interface/StoredProductProvenance.h

 Description: The per event per product provenance information to be stored

 Usage:
    This class has been optimized for storage

*/
//
// Original Author:  Chris Jones
//         Created:  Mon May 23 15:42:17 CDT 2011
//

// system include files
#include <vector>

// user include files

// forward declarations
namespace edm {
  struct StoredProductProvenance {
    StoredProductProvenance():branchID_(0), parentageIDIndex_(0) {}
    unsigned int branchID_;
    unsigned int parentageIDIndex_;
  };

  typedef std::vector<StoredProductProvenance> StoredProductProvenanceVector;

  inline
  bool
  operator<(StoredProductProvenance const& a, StoredProductProvenance const& b) {
    return a.branchID_ < b.branchID_;
  }
}

#endif
