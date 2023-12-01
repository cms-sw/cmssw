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
//

// user include files
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVectorBase.h"
#include "DataFormats/Common/interface/FillViewHelperVector.h"

// system include files
#include <type_traits>
#include <typeinfo>
#include <vector>

// forward declarations
namespace edm {
  class ProductID;
  template <typename T>
  class PtrVector;

  template <typename T>
  class PtrHolder {
  public:
    PtrHolder(Ptr<T> const& iPtr) : ptr_(iPtr) {}

    Ptr<T> const& operator*() const { return ptr_; }
    Ptr<T> const* operator->() const { return &ptr_; }

  private:
    Ptr<T> ptr_;
  };

  template <typename T>
  class PtrVectorItr {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Ptr<T>;
    using pointer = Ptr<T>*;
    using reference = Ptr<T> const;  // otherwise boost::range does not work
                                     // const, because this is a const_iterator
    using iterator = PtrVectorItr<T>;
    using difference_type = std::ptrdiff_t;

    PtrVectorItr(std::vector<void const*>::const_iterator const& iItr, PtrVector<T> const* iBase)
        : iter_(iItr), base_(iBase) {}

    Ptr<T> const operator*() const { return base_->fromItr(iter_); }

    Ptr<T> const operator[](difference_type n) const {  // Otherwise the
      return base_->fromItr(iter_ + n);                 // boost::range
    }                                                   // doesn't have []

    PtrHolder<T> operator->() const { return PtrHolder<T>(this->operator*()); }

    iterator& operator++() {
      ++iter_;
      return *this;
    }
    iterator& operator--() {
      --iter_;
      return *this;
    }
    iterator& operator+=(difference_type n) {
      iter_ += n;
      return *this;
    }
    iterator& operator-=(difference_type n) {
      iter_ -= n;
      return *this;
    }

    iterator operator++(int) {
      iterator it(*this);
      ++iter_;
      return it;
    }
    iterator operator--(int) {
      iterator it(*this);
      --iter_;
      return it;
    }
    iterator operator+(difference_type n) const {
      iterator it(*this);
      it.iter_ += n;
      return it;
    }
    iterator operator-(difference_type n) const {
      iterator it(*this);
      it.iter_ -= n;
      return it;
    }

    difference_type operator-(iterator const& rhs) const { return this->iter_ - rhs.iter_; }

    bool operator==(iterator const& rhs) const { return this->iter_ == rhs.iter_; }
    bool operator!=(iterator const& rhs) const { return this->iter_ != rhs.iter_; }
    bool operator<(iterator const& rhs) const { return this->iter_ < rhs.iter_; }
    bool operator>(iterator const& rhs) const { return this->iter_ > rhs.iter_; }
    bool operator<=(iterator const& rhs) const { return this->iter_ <= rhs.iter_; }
    bool operator>=(iterator const& rhs) const { return this->iter_ >= rhs.iter_; }

  private:
    std::vector<void const*>::const_iterator iter_;
    PtrVector<T> const* base_;
  };

  template <typename T>
  class PtrVector : public PtrVectorBase {
  public:
    using const_iterator = PtrVectorItr<T>;
    using iterator = PtrVectorItr<T>;  // make boost::sub_range happy (std allows this)
    using value_type = Ptr<T>;
    using member_type = T;
    using collection_type = void;

    friend class PtrVectorItr<T>;
    PtrVector() : PtrVectorBase() {}
    explicit PtrVector(ProductID const& iId) : PtrVectorBase(iId) {}
    PtrVector(PtrVector<T> const& iOther) : PtrVectorBase(iOther) {}

    template <typename U>
    PtrVector(PtrVector<U> const& iOther) : PtrVectorBase(iOther) {
      static_assert(std::is_base_of<T, U>::value, "PtrVector being copied is not of compatible type");
    }

    // ---------- const member functions ---------------------

    Ptr<T> operator[](unsigned long const iIndex) const { return this->makePtr<Ptr<T> >(iIndex); }

    const_iterator begin() const { return const_iterator(this->void_begin(), this); }

    const_iterator end() const { return const_iterator(this->void_end(), this); }
    // ---------- member functions ---------------------------

    void push_back(Ptr<T> const& iPtr) {
      this->push_back_base(
          iPtr.refCore(), iPtr.key(), iPtr.hasProductCache() ? iPtr.operator->() : static_cast<void const*>(nullptr));
    }

    template <typename U>
    void push_back(Ptr<U> const& iPtr) {
      //check that types are assignable
      static_assert(std::is_base_of<T, U>::value,
                    "Ptr used in push_back can not be converted to type used by PtrVector.");
      this->push_back_base(
          iPtr.refCore(), iPtr.key(), iPtr.hasProductCache() ? iPtr.operator->() : static_cast<void const*>(nullptr));
    }

    void swap(PtrVector& other) { this->PtrVectorBase::swap(other); }

    PtrVector& operator=(PtrVector const& rhs) {
      PtrVector temp(rhs);
      this->swap(temp);
      return *this;
    }

    void fillView(std::vector<void const*>& pointers, FillViewHelperVector& helpers) const;

    //Used by ROOT storage
    CMS_CLASS_VERSION(8)

  private:
    //PtrVector const& operator=(PtrVector const&); // stop default
    std::type_info const& typeInfo() const override { return typeid(T); }

    // ---------- member data --------------------------------
    Ptr<T> fromItr(std::vector<void const*>::const_iterator const& iItr) const { return this->makePtr<Ptr<T> >(iItr); }
  };

  template <typename T>
  void PtrVector<T>::fillView(std::vector<void const*>& pointers, FillViewHelperVector& helpers) const {
    pointers.reserve(this->size());
    for (const_iterator i = begin(), e = end(); i != e; ++i) {
      Ptr<T> ref = *i;
      T const* address = ref.isNull() ? nullptr : &*ref;
      pointers.push_back(address);
      helpers.push_back(FillViewHelperVector::value_type(ref.id(), ref.key()));
    }
  }

  template <typename T>
  inline void fillView(PtrVector<T> const& obj,
                       ProductID const&,
                       std::vector<void const*>& pointers,
                       FillViewHelperVector& helpers) {
    obj.fillView(pointers, helpers);
  }

  template <typename T>
  struct has_fillView<PtrVector<T> > {
    static bool const value = true;
  };

  // Free swap function
  template <typename T>
  inline void swap(PtrVector<T>& lhs, PtrVector<T>& rhs) {
    lhs.swap(rhs);
  }
}  // namespace edm
#endif
