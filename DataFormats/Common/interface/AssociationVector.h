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
 * \version $Revision: 1.8 $
 */

#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_same.hpp"

namespace edm {

  template<typename KeyRefProd, typename CVal, 
    typename KeyRef=edm::Ref<typename KeyRefProd::product_type>,
    typename SizeType=typename KeyRefProd::product_type::size_type>
  class AssociationVector {
  public:
    BOOST_STATIC_ASSERT( ( boost::is_same<SizeType, typename CVal::size_type>::value ) );
    typedef typename KeyRefProd::product_type CKey;
    typedef SizeType size_type;
    typedef typename KeyRef::value_type key_type;
    typedef typename CVal::value_type val_type;
    typedef typename std::pair<KeyRef, const val_type &> value_type;
    typedef typename std::pair<KeyRef, val_type> reference;
    typedef value_type const_reference;
    AssociationVector();
    AssociationVector(KeyRefProd ref);
    AssociationVector(const AssociationVector &);
    ~AssociationVector();
    
    size_type size() const;
    bool empty() const;
    reference operator[](size_type);
    const_reference operator[](size_type) const;
    
    AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & 
      operator=(const AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> &);
    
    void clear();
    void swap(AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & other);
    const KeyRefProd & keyProduct() const { return ref_; }
    KeyRef key(size_type i) const { return KeyRef(ref_, i); }
    const val_type & value(size_type i) const { return data_[ i ]; }  
    val_type & value(size_type i) { return data_[ i ]; }  
    void fillView(std::vector<void const*>& pointers) const;

  private:
    CVal data_;
    KeyRefProd ref_;
  };
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::AssociationVector() : 
    data_(), ref_() { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::AssociationVector(KeyRefProd ref) : 
    data_(ref->size()), ref_(ref) { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::AssociationVector(const AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & o) : 
    data_(o.data_), ref_(o.ref_) { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::~AssociationVector() { }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & 
  AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::operator=(const AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & o) {
    data_ = o.data_;
    ref_ = o.ref_;
    return * this;
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::size_type AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::size() const {
    return data_.size();
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline bool AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::empty() const {
    return data_.empty();
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::reference 
    AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::operator[](size_type n) {
    return reference( key(n), value(n) );
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline typename AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::const_reference 
    AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::operator[](size_type n) const {
    return const_reference( key(n), value(n) );
  }
  
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::clear() {
    data_.clear();
    ref_ = KeyRefProd();
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::swap(AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & other) {
    data_.swap(other.data_);
    std::swap(ref_, other.ref_);
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  void AssociationVector<KeyRefProd, CVal, KeyRef, SizeType>::fillView(std::vector<void const*>& pointers) const
  {
    pointers.reserve(this->size());
    for(typename CVal::const_iterator i=data_.begin(), e=data_.end(); i!=e; ++i)
      pointers.push_back(&(*i));
  }

  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline void swap(AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & a, 
		   AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> & b) {
    a.swap(b);
  }

  //----------------------------------------------------------------------
  //
  // Free function template to support creation of Views.

  template <typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  inline
  void
  fillView(AssociationVector<KeyRefProd,CVal, KeyRef, SizeType> const& obj,
	   std::vector<void const*>& pointers) {
    obj.fillView(pointers);
  }

  template <typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  struct has_fillView<edm::AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> > {
    static bool const value = true;
  };

#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template<typename KeyRefProd, typename CVal, typename KeyRef, typename SizeType>
  struct has_swap<edm::AssociationVector<KeyRefProd, CVal, KeyRef, SizeType> > {
    static bool const value = true;
  };
#endif

}

#endif
