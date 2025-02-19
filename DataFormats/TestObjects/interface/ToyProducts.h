#ifndef DataFormats_TestObjects_ToyProducts_h
#define DataFormats_TestObjects_ToyProducts_h

/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "FWCore/Utilities/interface/typedefs.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace edmtest {

  // Toy products

  struct ProductWithNoDictionary {
  };

  struct DummyProduct {
  };

  struct IntProduct {
    explicit IntProduct(int i = 0) : value(i) {}
    ~IntProduct() {}

    cms_int32_t value;
  };

  struct TransientIntProduct {
    explicit TransientIntProduct(int i = 0) : value(i) {}
    ~TransientIntProduct() {}

    cms_int32_t value;
    ProductWithNoDictionary dummy;
  };

  struct Int16_tProduct {
    explicit Int16_tProduct(int16_t i = 0, uint16_t j = 1) :value(i), uvalue(j) {}
    ~Int16_tProduct() {}
    int16_t value;
    uint16_t uvalue;
  };

  struct DoubleProduct {
    explicit DoubleProduct(double d = 2.2) : value(d) {}
    ~DoubleProduct() {}

    double value;
  };

  struct StringProduct {
    StringProduct() : name_() {}
    explicit StringProduct(std::string const& s) : name_(s) {}
    std::string name_;
  };

  struct Simple {
    Simple() : key(0), value(0.0) {}
    virtual ~Simple();
    typedef cms_int32_t key_type;
    key_type    key;
    double      value;
    key_type id() const { return key; }
    virtual Simple* clone() const;
  };

  inline
  bool
  operator==(Simple const& a, Simple const& b) {
    return(a.key == b.key && a.value == b.value);
  }

  inline
  bool operator<(Simple const& a, Simple const& b) {
    return a.key < b.key;
  }

  struct SimpleDerived : public Simple {
    SimpleDerived() : Simple(), dummy(0.0) {}
    virtual ~SimpleDerived();
    double dummy;
    virtual SimpleDerived* clone() const;
  };

  struct Sortable {
    cms_int32_t data;
    Sortable() : data(0) {}
    explicit Sortable(int i) : data(i) {}
  };

  inline
  bool
  operator==(Sortable const& a, Sortable const& b) {
    return(a.data == b.data);
  }

  inline
  bool operator<(Sortable const& a, Sortable const& b) {
    return a.data < b.data;
  }

  struct Unsortable : public edm::DoNotSortUponInsertion {
    cms_int32_t data;
    Unsortable() : data(0) {}
    explicit Unsortable(int i) : data(i) {}
  };

  inline
  bool operator<(Unsortable const&, Unsortable const&) {
    throw std::logic_error("operator< called for Unsortable");
  }

  struct Prodigal : public edm::DoNotRecordParents {
    cms_int32_t data;
    Prodigal() : data(0) {}
    explicit Prodigal(int i) : data(i) {}
  };

  typedef edm::SortedCollection<Simple> SCSimpleProduct;
  typedef std::vector<Simple>           VSimpleProduct;
  typedef edm::OwnVector<Simple>        OVSimpleProduct;
  typedef edm::OwnVector<SimpleDerived> OVSimpleDerivedProduct;
  typedef edm::AssociationVector<edm::RefProd<std::vector<Simple> >, std::vector<Simple> > AVSimpleProduct;
  typedef edm::DetSetVector<Sortable>   DSVSimpleProduct;
  typedef edm::DetSetVector<Unsortable> DSVWeirdProduct;

  typedef edmNew::DetSetVector<Sortable>      DSTVSimpleProduct;
  typedef edmNew::DetSetVector<SimpleDerived> DSTVSimpleDerivedProduct;
}
#endif
