#ifndef Common_RefToBaseVector_h
#define Common_RefToBaseVector_h
/**\class RefToBaseVector
 *
 * \author Luca Lista, INFN
 *
 */

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/IndirectVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "FWCore/Utilities/interface/EDMException.h"


namespace edm {
  template <class T>
  class RefToBaseVector {
  public:
    typedef RefToBase<T>                         value_type;
    typedef T                                    member_type;
    typedef reftobase::BaseVectorHolder<T>       holder_type;
    typedef typename holder_type::size_type      size_type;
    typedef typename holder_type::const_iterator const_iterator;

    RefToBaseVector();
    RefToBaseVector(RefToBaseVector const& iOther);
    template <class TRefVector> explicit RefToBaseVector(TRefVector const& iRef);

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
    const_iterator begin() const;
    const_iterator end() const;

  private:
    holder_type* holder_;
  };
  
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
  template <class TRefVector>
  inline
  RefToBaseVector<T>::RefToBaseVector(const TRefVector& iRef) :
    holder_(new reftobase::VectorHolder<T,TRefVector>(iRef)) 
  { }

  template <class T>
  inline
  RefToBaseVector<T>::RefToBaseVector(const RefToBaseVector<T>& iOther) : 
    holder_(iOther.holder_ ? iOther.holder_->clone() : 0)
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

//   template <class T>
//   inline
//   typename RefToBaseVector<T>::size_type
//   RefToBaseVector<T>::capacity() const 
//   {
//     return holder_ ? holder_->capacity() : 0;
//   }


//   template <class T>
//   inline
//   void 
//   RefToBaseVector<T>::reserve(size_type n)
//   {
//     if (!holder_) holder_ = new holder_type();
//     holder_->reserve(n);
//   }

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

}

#endif
