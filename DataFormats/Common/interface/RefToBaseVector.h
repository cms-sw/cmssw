#ifndef Common_RefToBaseVector_h
#define Common_RefToBaseVector_h
/**\class RefToBaseVector
 *
 * \author Luca Lista, INFN
 *
 * $Id: RefToBaseVector.h,v 1.10 2007/07/12 12:08:57 llista Exp $
 *
 */

#include "DataFormats/Provenance/interface/ProductID.h"
#include "boost/shared_ptr.hpp"

namespace edm {
  template<typename T> class RefToBase;
  class EDProductGetter;
  namespace reftobase {
    template<typename T> class BaseVectorHolder;
    class RefVectorHolderBase;
  }

  template <class T>
  class RefToBaseVector {
  public:
    typedef RefToBase<T>                         value_type;
    typedef T                                    member_type;
    typedef reftobase::BaseVectorHolder<T>       holder_type;
    typedef typename holder_type::size_type      size_type;
    typedef typename holder_type::const_iterator const_iterator;

    RefToBaseVector();
    RefToBaseVector(RefToBaseVector const& );
    template<class REFV> 
    explicit RefToBaseVector(REFV const& );
    RefToBaseVector(boost::shared_ptr<reftobase::RefVectorHolderBase> p);
    RefToBaseVector& operator=(RefToBaseVector const& iRHS);
    void swap(RefToBaseVector& other);

    ~RefToBaseVector();

    //void reserve(size_type n);
    void clear();

    value_type at(size_type idx) const;
    value_type operator[](size_type idx) const;
    bool isValid() const { return holder_ != 0; }
    bool isInvalid() const { return holder_ == 0; }
    bool empty() const;
    size_type size() const;
    //size_type capacity() const;
    ProductID id() const;
    EDProductGetter const * productGetter() const;
    const_iterator begin() const;
    const_iterator end() const;

    void push_back( const RefToBase<T> & );

    void fillView(std::vector<void const*>& pointers) const;

    std::auto_ptr<reftobase::RefVectorHolderBase> vectorHolder() const;

  private:
    holder_type * holder_;
  };
}

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/IndirectVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/traits.h"

namespace edm {  
  template <class T>
  inline
  void
  swap(RefToBaseVector<T>& a, RefToBaseVector<T>& b) {
    a.swap(b);
  }

  template <class T>
  inline
  bool
  operator== (RefToBaseVector<T> const& a,
	      RefToBaseVector<T> const& b)
  {
    if ( a.isInvalid() && b.isInvalid() ) return true;
    if ( a.isInvalid() || b.isInvalid() ) return false;
    return  a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
  }

  //--------------------------------------------------------------------
  // Implementation of RefToBaseVector<T>
  //--------------------------------------------------------------------
  
  template <class T>
  inline
  RefToBaseVector<T>::RefToBaseVector() : 
    holder_(0) 
  { }

  template <class T>
  template <class REFV>
  inline
  RefToBaseVector<T>::RefToBaseVector(const REFV& iRef) :
    holder_(new reftobase::VectorHolder<T,REFV>(iRef)) 
  { }

  template <class T>
  inline
  RefToBaseVector<T>::RefToBaseVector(const RefToBaseVector<T>& iOther) : 
    holder_(iOther.holder_ ? iOther.holder_->clone() : 0)
  { }

  template <class T>
  inline
  RefToBaseVector<T>::RefToBaseVector(boost::shared_ptr<reftobase::RefVectorHolderBase> p) : 
    holder_(new reftobase::IndirectVectorHolder<T>(p))
  { }

  template <class T>
  inline
  RefToBaseVector<T>& 
  RefToBaseVector<T>::operator=(const RefToBaseVector& iRHS) {
    RefToBaseVector temp(iRHS);
    this->swap(temp);
    return *this;
  }

  template <class T>
  inline
  void
  RefToBaseVector<T>::swap(RefToBaseVector& other) {
    std::swap(holder_, other.holder_);
  }

  template <class T>
  inline
  RefToBaseVector<T>::~RefToBaseVector() 
  {
    delete holder_; 
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::value_type
  RefToBaseVector<T>::at(size_type idx) const 
  {
    if ( holder_ == 0 )
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "Trying to dereference null RefToBaseVector<T> in method: at(" << idx  <<")";
    return holder_->at( idx );
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::value_type
  RefToBaseVector<T>::operator[](size_type idx) const 
  {
    return at( idx ); 
  }

  template <class T>
  inline
  bool 
  RefToBaseVector<T>::empty() const 
  {
    return holder_ ? holder_->empty() : true;
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::size_type
  RefToBaseVector<T>::size() const 
  {
    return holder_ ? holder_->size() : 0;
  }

  template <class T>
  inline
  void 
  RefToBaseVector<T>::clear()
  {
    if ( holder_ != 0 )
      holder_->clear();
  }

  template <class T>
  inline
  ProductID
  RefToBaseVector<T>::id() const
  {
    if ( holder_ == 0 )
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "Trying to dereference null RefToBaseVector<T> in method: id()";
    return holder_->id();
  }

  template <class T>
  inline
  EDProductGetter const * 
  RefToBaseVector<T>::productGetter() const
  {
    return holder_ ? holder_->productGetter() : 0;
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::const_iterator
  RefToBaseVector<T>::begin() const
  {
    if ( holder_ == 0 ) return const_iterator();
    return holder_->begin();
  }

  template <class T>
  inline
  typename RefToBaseVector<T>::const_iterator
  RefToBaseVector<T>::end() const
  {
    if ( holder_ == 0 ) return const_iterator();
    return holder_->end();
  }

  template <typename T>
  void
  RefToBaseVector<T>::fillView(std::vector<void const*>& pointers) const
  {
    typedef RefToBase<T>             ref_type;
    pointers.reserve(this->size());
    for (const_iterator i=begin(), e=end(); i!=e; ++i) {
      ref_type ref = * i;
      member_type const * address = ref.isNull() ? 0 : & * ref;
      pointers.push_back(address);
    }
  }

  // NOTE: the following implementation has unusual signature!
  template <typename T>
  inline void fillView(RefToBaseVector<T> const& obj,
		       std::vector<void const*>& pointers) {
    obj.fillView(pointers);
  }

  template <typename T>
  struct has_fillView<RefToBaseVector<T> > {
    static bool const value = true;
  };

  template <typename T>
  void RefToBaseVector<T>::push_back( const RefToBase<T> & r ) {
    if ( holder_ == 0 ) {
      std::auto_ptr<reftobase::BaseVectorHolder<T> > p = r.holder_->makeVectorHolder();
      holder_ = p.release();
    }
    holder_->push_back( r.holder_ );
  }

  template <typename T>
  std::auto_ptr<reftobase::RefVectorHolderBase> RefToBaseVector<T>::vectorHolder() const {
    return holder_->vectorHolder();
  }

}

#endif
