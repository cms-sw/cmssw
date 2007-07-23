#ifndef Common_AssociationVector_h
#define Common_AssociationVector_h
/* class edm::AssociationVector<CKey, CVal>
 *
 * adds to a std::vector<CVal> a edm::RefProd<CKey>, in such a way
 * that, assuming that the CVal and CKey collections have the same
 * size and are properly ordered, the two collections can be
 * in one-to-one correspondance
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.16.2.2 $
 */

#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_same.hpp"

namespace edm {

  template<typename KeyRefProd, typename CVal> 
  class AssociationVector {
    typedef AssociationVector<KeyRefProd, CVal> self;
    typedef size_t SizeType;
    typedef edm::Ref<typename KeyRefProd::product_type> KeyRef;

  public:
    typedef typename KeyRefProd::product_type CKey;
    typedef SizeType size_type;
    typedef typename KeyRef::value_type key_type;
    typedef typename std::pair<KeyRef, typename CVal::value_type> value_type;
    typedef std::vector<value_type> transient_vector_type;
    typedef const value_type & const_reference;
    AssociationVector();
    AssociationVector(KeyRefProd ref);
    AssociationVector(const AssociationVector &);
    ~AssociationVector();
    
    size_type size() const;
    bool empty() const;
    const_reference operator[](size_type n) const { fixup(); return transientVector_[ n ]; }
    const typename CVal::value_type & operator[](const KeyRef & k) const;
    typename CVal::value_type & operator[](const KeyRef & k);
    
    self & operator=(const self & );
    
    void clear();
    void swap( self & other);
    const KeyRefProd & keyProduct() const { return ref_; }

    KeyRef key(size_type i) const { return KeyRef(ref_, i); }
    const typename CVal::value_type & value(size_type i) const { return data_[ i ]; }
    void setValue(size_type i, const typename CVal::value_type & val ) { 
      data_[ i ] = val; 
      transientVector_[ i ].first = KeyRef(ref_, i);
      transientVector_[ i ].second = data_[ i ];
    }   

    typedef typename transient_vector_type::const_iterator const_iterator;

    const_iterator begin() const { fixup(); return transientVector_.begin(); } 
    const_iterator end() const { fixup(); return transientVector_.end(); } 

  private:
    CVal data_;
    KeyRefProd ref_;
    mutable transient_vector_type transientVector_;
    mutable bool fixed_;
    void fixup() const { 
      if (!fixed_) {
	fixed_ = true;
	transientVector_.resize( size() );
	for( size_type i = 0; i != size(); ++ i ) {
	  transientVector_[ i ] = make_pair( KeyRef(ref_, i), data_[ i ] );
	}
      }
    }
  };
  
  template<typename KeyRefProd, typename CVal>
  inline AssociationVector<KeyRefProd, CVal>::AssociationVector() : 
    data_(), ref_(), transientVector_(), fixed_(false)  { }
  
  template<typename KeyRefProd, typename CVal>
  inline AssociationVector<KeyRefProd, CVal>::AssociationVector(KeyRefProd ref) : 
    data_(ref->size()), ref_(ref), transientVector_(ref->size()), fixed_(true) { }
  
  template<typename KeyRefProd, typename CVal>
  inline AssociationVector<KeyRefProd, CVal>::
    AssociationVector(const AssociationVector<KeyRefProd, CVal> & o) : 
    data_(o.data_), ref_(o.ref_), transientVector_(o.transientVector_), fixed_(o.fixed_) { }
  
  template<typename KeyRefProd, typename CVal>
  inline AssociationVector<KeyRefProd, CVal>::~AssociationVector() { }
  
  template<typename KeyRefProd, typename CVal>
  inline AssociationVector<KeyRefProd, CVal> & 
  AssociationVector<KeyRefProd, CVal>::operator=(const self & o) {
    data_ = o.data_;
    ref_ = o.ref_;
    transientVector_ = o.transientVector_;
    fixed_ = o.fixed_;
    return * this;
  }
  
  template<typename KeyRefProd, typename CVal>
  inline typename AssociationVector<KeyRefProd, CVal>::size_type 
    AssociationVector<KeyRefProd, CVal>::size() const {
    return data_.size();
  }
  
  template<typename KeyRefProd, typename CVal>
  inline bool AssociationVector<KeyRefProd, CVal>::empty() const {
    return data_.empty();
  }
  
  template<typename KeyRefProd, typename CVal>
  inline const typename CVal::value_type &
  AssociationVector<KeyRefProd, CVal>::operator[]( const KeyRef & k ) const {
    if ( k.id() != ref_.id() )
      throw edm::Exception(edm::errors::InvalidReference) 
	<< "AssociationVector: trying to use [] operator passing a reference"
	<< " with the wrong product id (i.e.: pointing to the wrong collection)";
    return data_[ k.key() ];
  }

  template<typename KeyRefProd, typename CVal>
  inline typename CVal::value_type &
  AssociationVector<KeyRefProd, CVal>::operator[]( const KeyRef & k ) {
    if ( k.id() != ref_.id() )
      throw edm::Exception(edm::errors::InvalidReference) 
	<< "AssociationVector: trying to use [] operator passing a reference"
	<< " with the wrong product id (i.e.: pointing to the wrong collection)";
    return data_[ k.key() ];
  }

  template<typename KeyRefProd, typename CVal>
  inline void AssociationVector<KeyRefProd, CVal>::clear() {
    data_.clear();
    transientVector_.clear();
    ref_ = KeyRefProd();
    fixed_ = true;
  }

  template<typename KeyRefProd, typename CVal>
  inline void AssociationVector<KeyRefProd, CVal>::swap(self & other) {
    data_.swap(other.data_);
    transientVector_.swap(other.transientVector_);
    std::swap(ref_, other.ref_);
    std::swap(fixed_, other.fixed_);
  }

  template<typename KeyRefProd, typename CVal>
  inline void swap(AssociationVector<KeyRefProd, CVal> & a, 
		   AssociationVector<KeyRefProd, CVal> & b) {
    a.swap(b);
  }

#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template<typename KeyRefProd, typename CVal>
  struct has_swap<edm::AssociationVector<KeyRefProd, CVal> > {
    static bool const value = true;
  };
#endif

}

#endif
