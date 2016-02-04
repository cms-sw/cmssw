#ifndef DataFormats_Common_RefVectorIterator_h
#define DataFormats_Common_RefVectorIterator_h

/*----------------------------------------------------------------------
  
RefVectorIterator: An iterator for a RefVector
Note: this is actually a *const_iterator*

$Id: RefVectorIterator.h,v 1.10 2011/02/24 20:20:48 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/Ref.h"

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
    RefVectorIterator() : product_(), iter_() {}
    explicit RefVectorIterator(RefCore const& product, keyIter const& it) :
      product_(product), iter_(it) {}
    reference operator*() const {
      key_type const& key = *iter_;
      return value_type(product_, key);
    }
    reference operator[](difference n) const {
      key_type const& key = iter_[n];
      return value_type(product_, key);
    }
    std::auto_ptr<value_type> operator->() const {
      key_type const& key = *iter_;
      return std::auto_ptr<value_type>(new value_type(product_, key));
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

  private:
    RefCore product_;
    keyIter iter_;
  };

  template <typename C, typename T, typename F>
  inline
  RefVectorIterator<C, T, F> operator+(typename RefVectorIterator<C, T, F>::difference n, RefVectorIterator<C, T, F> const& iter) {
    return iter + n;
  } 
}
#endif
