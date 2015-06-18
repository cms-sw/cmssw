#ifndef DataFormats_Common_RefVectorIterator_h
#define DataFormats_Common_RefVectorIterator_h

/*----------------------------------------------------------------------
  
RefVectorIterator: An iterator for a RefVector
Note: this is actually a *const_iterator*


----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefTraits.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm {

  template <typename C, typename T = typename Ref<C>::value_type, typename F = typename Ref<C>::finder_type>
  class RefVectorIterator : public std::iterator <std::random_access_iterator_tag, Ref<C, T, F> > {
  public:
    typedef Ref<C, T, F> value_type;
    typedef Ref<C, T, F> const const_reference;  // Otherwise boost::iterator_reference assumes '*it' returns 'Ref &'
    typedef const_reference    reference;        // This to prevent compilation of code that tries to modify the RefVector
                                                 // through this iterator
    typedef typename value_type::key_type key_type;

    typedef RefVectorIterator<C, T, F> iterator;
    typedef std::ptrdiff_t difference;
    typedef typename std::vector<key_type>::const_iterator keyIter;
    typedef typename std::vector<void const*>::const_iterator MemberIter;

    RefVectorIterator() : refVector_(nullptr), nestedRefVector_(nullptr), iter_(0) {}

    explicit RefVectorIterator(RefVector<C, T, F> const* refVector,
                               typename RefVector<C, T, F>::size_type iter) :
      refVector_(refVector), nestedRefVector_(nullptr), iter_(iter) {}

    explicit RefVectorIterator(RefVector<RefVector<C, T, F>, T, typename refhelper::FindTrait<RefVector<C, T, F>, T>::value > const* refVector,
                             typename RefVector<C, T, F>::size_type iter) :
      refVector_(nullptr), nestedRefVector_(refVector), iter_(iter) {}

    reference operator*() const {
      if(refVector_) return (*refVector_)[iter_];
      return (*nestedRefVector_)[iter_];
    }
    reference operator[](difference n) const {
      typename RefVector<C, T, F>::size_type j = iter_ + n;
      if(refVector_) return (*refVector_)[j];
      return (*nestedRefVector_)[j];
    }

    class RefProxy {
    public:
      RefProxy(value_type const& ref) : ref_(ref) { }
      value_type const* operator->() const { return &ref_; }
    private:
      value_type ref_;
    };

    RefProxy operator->() const {
      if(refVector_) return RefProxy(value_type((*refVector_)[iter_]));
      return RefProxy(value_type((*nestedRefVector_)[iter_]));
    }
    iterator & operator++() {++iter_; return *this;}
    iterator & operator--() {--iter_; return *this;}
    iterator & operator+=(difference n) {iter_ += n; return *this;}
    iterator & operator-=(difference n) {iter_ -= n; return *this;}

    iterator operator++(int) {iterator it(*this); ++iter_; return it;}
    iterator operator--(int) {iterator it(*this); --iter_; return it;}
    iterator operator+(difference n) const {iterator it(*this); it.iter_+=n; return it;}
    iterator operator-(difference n) const {iterator it(*this); it.iter_-=n; return it;}

    difference operator-(iterator const& rhs) const {return this->iter_ - rhs.iter_;}

    bool operator==(iterator const& rhs) const {return this->iter_ == rhs.iter_;}
    bool operator!=(iterator const& rhs) const {return this->iter_ != rhs.iter_;}
    bool operator<(iterator const& rhs) const {return this->iter_ < rhs.iter_;}
    bool operator>(iterator const& rhs) const {return this->iter_ > rhs.iter_;}
    bool operator<=(iterator const& rhs) const {return this->iter_ <= rhs.iter_;}
    bool operator>=(iterator const& rhs) const {return this->iter_ >= rhs.iter_;}

    key_type  key() const {
      if(refVector_) return (*refVector_)[iter_].key();
      return (*nestedRefVector_)[iter_].key();
    }

  private:
    RefVector<C, T, F> const* refVector_;
    RefVector<RefVector<C, T, F>, T, typename refhelper::FindTrait<RefVector<C, T, F>, T>::value > const* nestedRefVector_;
    typename RefVector<C, T, F>::size_type iter_;
  };

  template <typename C, typename T, typename F>
  inline
  RefVectorIterator<C, T, F> operator+(typename RefVectorIterator<C, T, F>::difference n, RefVectorIterator<C, T, F> const& iter) {
    return iter + n;
  } 
}
#endif
