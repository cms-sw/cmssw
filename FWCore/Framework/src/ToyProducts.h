#ifndef EDM_TEST_TOYPRODUCTS_H
#define EDM_TEST_TOYPRODUCTS_H

/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>

namespace edmtest
{

  // Toy products

  struct DummyProduct
  { };

  struct IntProduct
  {
    explicit IntProduct(int i=0) : value(i) { }
    ~IntProduct() { }
    
    int value;
  };

  struct DoubleProduct
  {
    explicit DoubleProduct(double d=2.2) : value(d) { }
    ~DoubleProduct() { }
    
    double value;
  };

  struct StringProduct
  {
    StringProduct(){}
    explicit StringProduct(const std::string& s):name_(s){}
    std::string name_;
  };

}
#endif 
