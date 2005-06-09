#ifndef EDM_TEST_TOYPRODUCTS_H
#define EDM_TEST_TOYPRODUCTS_H

/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>

#include "FWCore/EDProduct/interface/EDProduct.h"

namespace edmtest
{

  // Toy products

  struct DummyProduct : public edm::EDProduct
  { };

  struct IntProduct : public edm::EDProduct
  {
    explicit IntProduct(int i=0) : value(i) { }
    ~IntProduct() { }
    
    int value;
  };

  struct DoubleProduct : public edm::EDProduct
  {
    explicit DoubleProduct(double d=2.2) : value(d) { }
    ~DoubleProduct() { }
    
    double value;
  };

  struct StringProduct :  public edm::EDProduct
  {
    explicit StringProduct(const std::string& s):name_(s){}
    std::string name_;
  };

}
#endif 
