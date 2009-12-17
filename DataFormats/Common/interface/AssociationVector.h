#ifndef DataFormats_Common_AssociationVector_h
#define DataFormats_Common_AssociationVector_h
/* class edm::AssociationVector<CKey, CVal>
 *
 * adds to a std::vector<CVal> a edm::RefProd<CKey>, in such a way
 * that, assuming that the CVal and CKey collections have the same
 * size and are properly ordered, the two collections can be
 * in one-to-one correspondance
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.28 $
 */

#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/FillView.h"

#include "DataFormats/Provenance/interface/ProductID.h"

#include "boost/static_assert.hpp"
#include "boost/type_traits/is_same.hpp"

namespace edm {
  namespace helper {

    struct AssociationIdenticalKeyReference {
      template<typename T>
      static const T & get( const T & t, edm::ProductID ) { return t; }
    };
    
    template<typename T>
    struct AssociationKeyReferenceTrait {
      typedef AssociationIdenticalKeyReference type;
    };

    template<typename REFPROD>
    struct RefFromRefProdTrait { };

    template<typename C>
    struct RefFromRefProdTrait<RefProd<C> > {
      typedef Ref<typename RefProd<C>::product_type> ref_type;
    };

    template<typename T>
    struct RefFromRefProdTrait<RefToBaseProd<T> > {
      typedef RefToBase<T> ref_type;
    };
  }

  template<typename KeyRefProd, typename CVal, 
    typename KeyRef = typename helper::RefFromRefProdTrait<KeyRefProd>::ref_type,
    typename SizeType = typename KeyRefProd::product_type::size_type,
    typename KeyReferenceHelper = typename helper::AssociationKeyReferenceTrait<KeyRef>::type>
  class AssociationVector {
    BOOST_STATIC_ASSERT( ( boost::is_convertible<SizeType, typename CVal::size_type>::value ) );
    typedef AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> self;

  public:
    typedef KeyRefProd refprod_type;
    typedef typename KeyRefProd::product_type CKey;
    typedef SizeType size_type;
    typedef typename KeyRef::value_type key_type;
    typedef typename std::pair<KeyRef, typename CVal::value_type> value_type;
    typedef std::vector<value_type> transient_vector_type;
    typedef const value_type & const_reference;
    AssociationVector();
    AssociationVector(const KeyRefProd & ref, const CKey * = 0);
    AssociationVector(const AssociationVector &);
    ~AssociationVector();
    
    size_type size() const;
    bool empty() const;
    const_reference operator[](size_type n) const;
    typename CVal::const_reference operator[](const KeyRef & k) const;
    typename CVal::reference operator[](const KeyRef & k);
    
    self & operator=(const self & );
    
    void clear();
    void swap(self & other);
    const KeyRefProd & keyProduct() const { return ref_; }
    KeyRef key(size_type i) const { return KeyRef(ref_, i); }
    const typename CVal::value_type value(size_type i) const { return data_[ i ]; }
    void setValue(size_type i, const typename CVal::value_type & val );
    void fillView(ProductID const& id,
		  std::vector<void const*>& pointers,
		  helper_vector& helpers) const;

    typedef typename transient_vector_type::const_iterator const_iterator;

    const_iterator begin() const { return transientVector().begin(); } 
    const_iterator end() const { return transientVector().end(); } 

  private:
    CVal data_;
    KeyRefProd ref_;
    mutable transient_vector_type transientVector_;
    mutable bool fixed_;
    const transient_vector_type & transientVector() const { fixup(); return transientVector_; }
    void fixup() const;
  };
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::AssociationVector() : 
    data_(), ref_(), transientVector_(), fixed_(false)  { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::AssociationVector(const KeyRefProd & ref,
												      const CKey * coll) : 
    data_(coll == 0 ? ref->size() : coll->size()), ref_(ref), 
    transientVector_(coll == 0 ? ref->size() : coll->size()), fixed_(true) { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::
    AssociationVector(const AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> & o) : 
    data_(o.data_), ref_(o.ref_), transientVector_(o.transientVector_), fixed_(o.fixed_) { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::~AssociationVector() { }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::const_reference 
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator[](size_type n) const { 
    return transientVector()[ n ]; 
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename CVal::const_reference
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator[]( const KeyRef & k ) const {
    KeyRef keyRef = KeyReferenceHelper::get( k, ref_.id() );
    if ( keyRef.id() == ref_.id() ) 
      return data_[ keyRef.key() ];
    else 
      throw edm::Exception(edm::errors::InvalidReference) 
	<< "AssociationVector: trying to use [] operator passing a reference"
	<< " with the wrong product id (i.e.: pointing to the wrong collection)";
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename CVal::reference
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator[]( const KeyRef & k ) {
    KeyRef keyRef = KeyReferenceHelper::get( k, ref_.id() );
    fixed_ = false;
    if ( keyRef.id() == ref_.id() ) 
      return data_[ keyRef.key() ];
    else 
      throw edm::Exception(edm::errors::InvalidReference) 
	<< "AssociationVector: trying to use [] operator passing a reference"
	<< " with the wrong product id (i.e.: pointing to the wrong collection)";
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> & 
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator=(const self & o) {
    data_ = o.data_;
    ref_ = o.ref_;
    fixed_ = false;
    return * this;
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::setValue(size_type i, const typename CVal::value_type & val ) { 
    data_[ i ] = val; 
    KeyRef ref(ref_, i);
    transientVector_[ i ].first = ref;
    transientVector_[ i ].second = data_[ i ];
  }  
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::size_type 
    AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::size() const {
    return data_.size();
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline bool AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::empty() const {
    return data_.empty();
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::clear() {
    data_.clear();
    transientVector_.clear();
    ref_ = KeyRefProd();
    fixed_ = true;
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::swap(self & other) {
    data_.swap(other.data_);
    transientVector_.swap(other.transientVector_);
    ref_.swap( other.ref_);
    std::swap(fixed_, other.fixed_);
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::fixup() const { 
    if (!fixed_) {
      fixed_ = true;
      transientVector_.resize( size() );
      for( size_type i = 0; i != size(); ++ i ) {
	transientVector_[ i ] = std::make_pair( KeyRef(ref_, i), data_[ i ] );
      }
    }
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::fillView(ProductID const& id,
											  std::vector<void const*>& pointers, 
											  helper_vector& helpers) const
  {
    detail::reallyFillView(*this, id, pointers, helpers);
//     pointers.reserve(this->size());
//     for(typename CVal::const_iterator i=data_.begin(), e=data_.end(); i!=e; ++i)
//       pointers.push_back(&(*i));
//     // helpers is not yet filled in.
//     //throw edm::Exception(errors::UnimplementedFeature, "AssociationVector<T>::fillView(...)");
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void swap(AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> & a, 
		   AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> & b) {
    a.swap(b);
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline
  void
  fillView(AssociationVector<KeyRefProd,CVal, KeyRef, SizeType, KeyReferenceHelper> const& obj,
	   ProductID const& id,
	   std::vector<void const*>& pointers,
	   helper_vector& helpers) {
    obj.fillView(id, pointers, helpers);
  }

  template <typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  struct has_fillView<edm::AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> > {
    static bool const value = true;
  };

}

#endif
