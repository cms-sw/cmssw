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
 */

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/CommonExceptions.h"
#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif
#include <memory>
#include "boost/static_assert.hpp"

namespace edm {
  namespace helper {

    struct AssociationIdenticalKeyReference {
      template<typename T>
      static T const& get(T const& t, ProductID) { return t; }
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
    typename SizeType = unsigned int,//the type used here can not change when go from 32bit to 64bit or across platforms
    typename KeyReferenceHelper = typename helper::AssociationKeyReferenceTrait<KeyRef>::type>
  class AssociationVector {
    BOOST_STATIC_ASSERT((boost::is_convertible<SizeType, typename CVal::size_type>::value));
    typedef AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> self;

  public:
    typedef KeyRefProd refprod_type;
    typedef typename KeyRefProd::product_type CKey;
    typedef SizeType size_type;
    typedef typename KeyRef::value_type key_type;
    typedef typename std::pair<KeyRef, typename CVal::value_type> value_type;
    typedef std::vector<value_type> transient_vector_type;
    typedef value_type const& const_reference;
    AssociationVector();
    AssociationVector(KeyRefProd const& ref, CKey const* = 0);
    AssociationVector(AssociationVector const&);
    ~AssociationVector();

    size_type size() const;
    bool empty() const;
    const_reference operator[](size_type n) const;
    typename CVal::const_reference operator[](KeyRef const& k) const;
    typename CVal::reference operator[](KeyRef const& k);

    self& operator=(self const&);

    void clear();
    void swap(self& other);
    KeyRefProd const& keyProduct() const { return ref_; }
    KeyRef key(size_type i) const { return KeyRef(ref_, i); }
    typename CVal::value_type const value(size_type i) const { return data_[ i ]; }
    void setValue(size_type i, typename CVal::value_type const& val);
    void fillView(ProductID const& id,
		  std::vector<void const*>& pointers,
		  helper_vector& helpers) const;

    typedef typename transient_vector_type::const_iterator const_iterator;

    const_iterator begin() const { return transientVector().begin(); }
    const_iterator end() const { return transientVector().end(); }

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    enum CacheState { kUnset, kFilling, kSet };
    CVal data_;
    KeyRefProd ref_;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    mutable std::atomic<transient_vector_type*> transientVector_;
#else
    mutable transient_vector_type* transientVector_;
#endif

    transient_vector_type const& transientVector() const;
    void fixup() const;
  };

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::transient_vector_type const&
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::transientVector() const {
    fixup();
    return *(transientVector_.load(std::memory_order_acquire)); }
#endif
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::AssociationVector() :
    data_(), ref_(), transientVector_(nullptr)  { }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::AssociationVector(KeyRefProd const& ref,
												      CKey const* coll) :
    data_(coll == 0 ? ref->size() : coll->size()), ref_(ref),
    transientVector_( new transient_vector_type(coll == 0 ? ref->size() : coll->size())) { }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::
    AssociationVector(AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> const& o) :
    data_(o.data_), ref_(o.ref_), transientVector_() {
      auto t = o.transientVector_.load(std::memory_order_acquire);
      if(t) {
        transientVector_.store( new transient_vector_type(*t), std::memory_order_release);
      }
    }
#endif
  
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::~AssociationVector() {
    delete transientVector_.load(std::memory_order_acquire);
  }
#endif

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::const_reference
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator[](size_type n) const {
    return transientVector()[ n ];
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename CVal::const_reference
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator[](KeyRef const& k) const {
    KeyRef keyRef = KeyReferenceHelper::get(k, ref_.id());
    checkForWrongProduct(keyRef.id(), ref_.id());
    return data_[ keyRef.key() ];
  }


#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename CVal::reference
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator[](KeyRef const& k) {
    KeyRef keyRef = KeyReferenceHelper::get(k, ref_.id());
    auto t = transientVector_.exchange(nullptr,std::memory_order_acq_rel);
    delete t;
    checkForWrongProduct(keyRef.id(), ref_.id());
    return data_[ keyRef.key() ];
  }
#endif
  
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>&
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::operator=(self const& o) {
    if(this == &o) {
      return *this;
    }
    data_ = o.data_;
    ref_ = o.ref_;
    auto t =transientVector_.exchange(nullptr, std::memory_order_acq_rel);
    delete t;
    return *this;
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::setValue(size_type i, typename CVal::value_type const& val) {
    data_[ i ] = val;
    KeyRef ref(ref_, i);
    auto t = transientVector_.load(std::memory_order_acquire);
    (*t)[ i ].first = ref;
    (*t)[ i ].second = data_[ i ];
  }
#endif

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::size_type
    AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::size() const {
    return data_.size();
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline bool AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::empty() const {
    return data_.empty();
  }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::clear() {
    data_.clear();
    auto t = transientVector_.load(std::memory_order_acquire);
    if(t) t->clear();
    ref_ = KeyRefProd();
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::swap(self& other) {
    data_.swap(other.data_);
    other.transientVector_.store(transientVector_.exchange(other.transientVector_.load(std::memory_order_acquire),std::memory_order_acq_rel),std::memory_order_release);
    ref_.swap(other.ref_);
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>::fixup() const {
    if (nullptr == transientVector_.load(std::memory_order_acquire)) {
      std::unique_ptr<transient_vector_type> newT {new transient_vector_type(size()) };
      for(size_type i = 0; i != size(); ++i) {
        (*newT)[ i ] = std::make_pair(KeyRef(ref_, i), data_[ i ]);
      }
      transient_vector_type* expected = nullptr;
      if(transientVector_.compare_exchange_strong(expected, newT.get()) ) {
        newT.release();
      }
    }
  }
#endif

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
//     //Exception::throwThis(errors::UnimplementedFeature, "AssociationVector<T>::fillView(...)");
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType, typename KeyReferenceHelper>
  inline void swap(AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>& a,
		   AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper>& b) {
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
  struct has_fillView<AssociationVector<KeyRefProd, CVal, KeyRef, SizeType, KeyReferenceHelper> > {
    static bool const value = true;
  };

}

#endif
