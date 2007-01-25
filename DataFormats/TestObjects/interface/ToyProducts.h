#ifndef TestObjects_ToyProducts_h
#define TestObjects_ToyProducts_h

/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"

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
    StringProduct() : name_() {}
    explicit StringProduct(const std::string& s) : name_(s){}
    std::string name_;
  };

  struct Simple
  {
    Simple() : key(0), value(0.0) { }
    Simple(const Simple& in) : key(in.key), value(in.value) { }
    typedef int key_type;
    key_type    key;
    double      value;
    key_type id() const { return key; }
    Simple* clone() const { return new Simple(*this); }
  };

  inline
  bool
  operator== (Simple const& a, Simple const& b)
  {
    return (a.key==b.key && a.value==b.value);
  }
  
  inline
  bool operator< (Simple const& a, Simple const& b)
  {
    return a.key < b.key;
  }

  struct Sortable
  {
    int data;
    Sortable() : data(0) { }
    explicit Sortable(int i) : data(i) { }
  };

  inline
  bool operator< (Sortable const& a, Sortable const& b)
  {
    return a.data < b.data;
  }

  struct Unsortable : public edm::DoNotSortUponInsertion
  {
    int data;
    Unsortable() : data(0) { }
    explicit Unsortable(int i) : data(i) { }
  };

  inline
  bool operator< (Unsortable const&, Unsortable const&)
  {
    throw std::logic_error("operator< called for Unsortable");
  }


    

  typedef edm::SortedCollection<Simple> SCSimpleProduct;
  typedef edm::OwnVector<Simple>        OVSimpleProduct;
  typedef edm::DetSetVector<Sortable>   DSVSimpleProduct;
  typedef edm::DetSetVector<Unsortable> DSVWeirdProduct;

}
#endif
