// -*- C++ -*-
//
// Package:     EDProduct
// Class  :     EDProductGetter
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov  1 15:06:41 EST 2005
//

// system include files

// user include files
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  //
  // constants, enums and typedefs
  //
  
  //
  // static data member definitions
  //
  
  //
  // constructors and destructor
  //
  EDProductGetter::EDProductGetter()
  {
  }
  
  // EDProductGetter::EDProductGetter(EDProductGetter const& rhs)
  // {
  //    // do actual copying here;
  // }
  
  EDProductGetter::~EDProductGetter()
  {
  }
  
  //
  // assignment operators
  //
  // EDProductGetter const& EDProductGetter::operator=(EDProductGetter const& rhs)
  // {
  //   //An exception safe implementation is
  //   EDProductGetter temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }
  
  //
  // member functions
  //
  
  //
  // const member functions
  //
  
  //
  // static member functions
  //
  
  EDProductGetter const*
  mustBeNonZero(EDProductGetter const* prodGetter, std::string refType, ProductID const& productID) {
    if (prodGetter != 0) return prodGetter;
        throw Exception(errors::InvalidReference, refType)
  	<< "Attempt to construct a " << refType << " with ProductID " << productID << "\n"
  	<< "but with a null pointer to a product getter.\n"
  	<< "The product getter pointer passed to the constructor must refer\n"
  	<< "to a real getter, such as an EventPrincipal.\n";
  }
  
  static EDProductGetter const* s_productGetter=0;
  EDProductGetter const* 
  EDProductGetter::switchProductGetter(EDProductGetter const* iNew) 
  {
    //std::cout <<"switch from "<<s_productGetter<<" to "<<iNew<<std::endl;
    EDProductGetter const* old = s_productGetter;
    s_productGetter = iNew;
    return old;
  }
  void 
  EDProductGetter::assignEDProductGetter(EDProductGetter const* & iGetter)
  {    
    //std::cout <<"assign "<<s_productGetter<<std::endl;
    
    iGetter = s_productGetter;
  }


}
