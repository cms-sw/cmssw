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
// $Id: PtrVector.h,v 1.3 2008/02/15 05:57:03 wmtan Exp $
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
    PtrHolder(Ptr<T> const& iPtr) :ptr_(iPtr) {}
    
    Ptr<T> const& operator*() const {
      return ptr_;
    }
    Ptr<T> const* operator->() const {
      return &ptr_;
    }
  private:
    Ptr<T> ptr_;
  };
  
  template <typename T>
  class PtrVectorItr : public std::iterator <std::random_access_iterator_tag, Ptr<T> > {
  public:
    typedef Ptr<T> const reference; // otherwise boost::range does not work
                                    // const, because this is a const_iterator
    typedef PtrVectorItr<T> iterator;
    typedef typename std::iterator <std::random_access_iterator_tag, Ptr<T> >::difference_type difference_type;
    
    PtrVectorItr(std::vector<void const*>::const_iterator const& iItr,
                 PtrVector<T> const* iBase):
    iter_(iItr),
    base_(iBase) {}
    
    Ptr<T> const operator*() const {
      return base_->fromItr(iter_);
    }

    Ptr<T> const operator[](difference_type n) const {  // Otherwise the
      return base_->fromItr(iter_+n);        // boost::range 
    }                                        // doesn't have []

    
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
    std::vector<void const*>::const_iterator iter_;
    PtrVector<T> const* base_;
  };
  
  template <typename T>
  class PtrVector : public PtrVectorBase {
    
  public:
    
    typedef PtrVectorItr<T> const_iterator;
    typedef PtrVectorItr<T> iterator; // make boost::sub_range happy (std allows this)
    typedef Ptr<T> value_type;
    
    friend class PtrVectorItr<T>;
    PtrVector() {}
    PtrVector(PtrVector<T> const& iOther): PtrVectorBase(iOther) {}
    
    template <typename U>
    PtrVector(PtrVector<U> const& iOther): PtrVectorBase(iOther) {
      BOOST_STATIC_ASSERT( (boost::is_base_of<T, U>::value) );
    }
    
    // ---------- const member functions ---------------------
    
    Ptr<T> operator[](unsigned long const iIndex ) const {
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
    
    void push_back(Ptr<T> const& iPtr) {
      this->push_back_base(iPtr.refCore(),
			   iPtr.key(),
			   iPtr.hasCache() ? iPtr.operator->() : static_cast<void const*>(0));
    }

    template<typename U>
    void push_back(Ptr<U> const& iPtr) {
      //check that types are assignable
      BOOST_STATIC_ASSERT( (boost::is_base_of<T, U>::value) );
      this->push_back_base(iPtr.refCore(),
			   iPtr.key(),
			   iPtr.hasCache() ? iPtr.operator->() : static_cast<void const*>(0));
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
    
    //PtrVector const& operator=(PtrVector const&); // stop default
    std::type_info const& typeInfo() const {return typeid(T);}

    // ---------- member data --------------------------------
    Ptr<T> fromItr(std::vector<void const*>::const_iterator const& iItr) const {
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
