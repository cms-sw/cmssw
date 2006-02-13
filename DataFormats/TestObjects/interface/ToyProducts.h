#ifndef TestObjects_ToyProducts_h
#define TestObjects_ToyProducts_h

/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>

#include "DataFormats/Common/interface/SortedCollection.h"

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

  struct Simple
  {
    typedef int key_type;
    key_type    key;
    double      value;
    key_type id() const { return key; }
  };

  typedef edm::SortedCollection<Simple> SCSimpleProduct;

}
#endif
