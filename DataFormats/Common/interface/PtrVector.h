#ifndef DataFormats_Common_PtrVector_h
#define DataFormats_Common_PtrVector_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     PtrVector
// 
/**\class PtrVector PtrVector.h DataFormats/Common/interface/PtrVector.h

 Description: A container which returns edm::Ptr<>'s referring to items in one container in the edm::Event

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Oct 24 15:26:50 EDT 2007
// $Id: PtrVector.h,v 1.1 2007/11/06 20:17:45 chrjones Exp $
//

// system include files
#include "boost/iterator.hpp"
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_base_of.hpp"
#include <vector>

// user include files
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVectorBase.h"

// forward declarations
namespace edm {
  template <typename T> class PtrVector;
  
  template <typename T>
  class PtrHolder {
  public:
    PtrHolder( const Ptr<T>& iPtr) :ptr_(iPtr) {}
    
    const Ptr<T>& operator*() const {
      return ptr_;
    }
    const Ptr<T>* operator->() const {
      return &ptr_;
    }
  private:
    Ptr<T> ptr_;
  };
  
  template <typename T>
  class PtrVectorItr : public std::iterator <std::random_access_iterator_tag, Ptr<T> > {
  public:
    typedef PtrVectorItr<T> iterator;
    typedef typename std::iterator <std::random_access_iterator_tag, Ptr<T> >::difference_type difference_type;
    
    PtrVectorItr(const std::vector<const void*>::const_iterator& iItr,
                 const PtrVector<T>* iBase):
    iter_(iItr),
    base_(iBase) {}
    
    Ptr<T> operator*() const {
      return base_->fromItr(iter_);
    }
    
    PtrHolder<T> operator->() const {
      return PtrHolder<T>( this->operator*() );
    }
    
    iterator & operator++() {++iter_; return *this;}
    iterator & operator--() {--iter_; return *this;}
    iterator & operator+=(difference_type n) {iter_ += n; return *this;}
    iterator & operator-=(difference_type n) {iter_ -= n; return *this;}
    
    iterator operator++(int) {iterator it(*this); ++iter_; return it;}
    iterator operator--(int) {iterator it(*this); --iter_; return it;}
    iterator operator+(difference_type n) const {iterator it(*this); it.iter_+=n; return it;}
    iterator operator-(difference_type n) const {iterator it(*this); it.iter_-=n; return it;}
    
    difference_type operator-(iterator const& rhs) const {return this->iter_ - rhs.iter_;}
    
    bool operator==(iterator const& rhs) const {return this->iter_ == rhs.iter_;}
    bool operator!=(iterator const& rhs) const {return this->iter_ != rhs.iter_;}
    bool operator<(iterator const& rhs) const {return this->iter_ < rhs.iter_;}
    bool operator>(iterator const& rhs) const {return this->iter_ > rhs.iter_;}
    bool operator<=(iterator const& rhs) const {return this->iter_ <= rhs.iter_;}
    bool operator>=(iterator const& rhs) const {return this->iter_ >= rhs.iter_;}
    
  private:
    std::vector<const void*>::const_iterator iter_;
    const PtrVector<T>* base_;
  };
  
  template <typename T>
  class PtrVector : public PtrVectorBase {
    
  public:
    
    typedef PtrVectorItr<T> const_iterator;
    typedef Ptr<T> value_type;
    
    friend class PtrVectorItr<T>;
    PtrVector() {}
    PtrVector(const PtrVector<T>& iOther): PtrVectorBase(iOther) {}
    
    template <typename U>
    PtrVector(const PtrVector<U>& iOther): PtrVectorBase(iOther) {
      BOOST_STATIC_ASSERT( (boost::is_base_of<T, U>::value) );
    }
    
    // ---------- const member functions ---------------------
    
    Ptr<T> operator[](const unsigned long iIndex ) const {
      return this->makePtr<Ptr<T> >(iIndex);
    }
    
    const_iterator begin() const {
      return const_iterator(this->void_begin(),
                            this);
    }
    
    const_iterator end() const {
      return const_iterator(this->void_end(),
                            this);
    }
    // ---------- member functions ---------------------------
    
    void push_back(const Ptr<T>& iPtr) {
      this->push_back_base(iPtr.key(),
                           iPtr.hasCache()? iPtr.operator->() : static_cast<const void*>(0),
                           iPtr.id(),
                           iPtr.productGetter());
    }

    template<typename U>
    void push_back(const Ptr<U>& iPtr) {
      //check that types are assignable
      BOOST_STATIC_ASSERT( (boost::is_base_of<T, U>::value) );
      this->push_back_base(iPtr.key(),
                           iPtr.hasCache()? iPtr.operator->() : static_cast<const void*>(0),
                           iPtr.id(),
                           iPtr.productGetter());
    }

    void swap(PtrVector& other) {
      this->PtrVectorBase::swap(other);
    }
    
    PtrVector& operator=(PtrVector const& rhs) {
      PtrVector temp(rhs);
      this->swap(temp);
      return *this;
    }

  private:
    
    //const PtrVector& operator=(const PtrVector&); // stop default
    const std::type_info& typeInfo() const {return typeid(T);}

    // ---------- member data --------------------------------
    Ptr<T> fromItr(const std::vector<const void*>::const_iterator& iItr) const {
      return this->makePtr<Ptr<T> >(iItr);
    }
    
  };
  
  // Free swap function
  template <typename T>
  inline
  void
  swap(PtrVector<T>& lhs, PtrVector<T>& rhs) {
    lhs.swap(rhs);
  }
}
#endif
