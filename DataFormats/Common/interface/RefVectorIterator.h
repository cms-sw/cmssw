#ifndef DataFormats_Common_RefVectorIterator_h
#define DataFormats_Common_RefVectorIterator_h

/*----------------------------------------------------------------------
  
RefVectorIterator: An iterator for a RefVector
Note: this is actually a *const_iterator*

$Id: RefVectorIterator.h,v 1.8 2007/12/21 22:42:30 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Common/interface/RefItem.h"
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
    typedef typename std::vector<RefItem<key_type> >::const_iterator itemIter;
    RefVectorIterator() : product_(), iter_() {}
    explicit RefVectorIterator(RefCore const& product, itemIter const& it) :
      product_(product), iter_(it) {}
    reference operator*() const {
      RefItem<key_type> const& item = *iter_;
      return value_type(product_, item);
    }
    reference operator[](difference n) const {
      RefItem<key_type> const& item = iter_[n];
      return value_type(product_, item);
    }
    std::auto_ptr<value_type> operator->() const {
      RefItem<key_type> const& item = *iter_;
      return std::auto_ptr<value_type>(new value_type(product_, item));
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
    itemIter iter_;
  };

  template <typename C, typename T, typename F>
  inline
  RefVectorIterator<C, T, F> operator+(typename RefVectorIterator<C, T, F>::difference n, RefVectorIterator<C, T, F> const& iter) {
    return iter + n;
  } 
}
#endif
