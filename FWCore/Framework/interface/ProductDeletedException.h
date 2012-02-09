#ifndef FWCore_Framework_ProductDeletedException_h
#define FWCore_Framework_ProductDeletedException_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProductDeletedException
// 
/**\class ProductDeletedException ProductDeletedException.h FWCore/Framework/interface/ProductDeletedException.h

 Description: Exception thrown if attempt to get data that has been deleted

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 12 13:32:29 CST 2012
// $Id$
//

// system include files

// user include files
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
  class ProductDeletedException : public cms::Exception {
    
  public:
    ProductDeletedException();
    //virtual ~ProductDeletedException();
    
    // ---------- const member functions ---------------------
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    
  private:
    //ProductDeletedException(const ProductDeletedException&); // stop default
    
    //const ProductDeletedException& operator=(const ProductDeletedException&); // stop default
    
    // ---------- member data --------------------------------
    
  };
  
}
#endif
