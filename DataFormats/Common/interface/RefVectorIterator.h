#ifndef Common_RefVectorIterator_h
#define Common_RefVectorIterator_h

/*----------------------------------------------------------------------
  
RefVectorIterator: An iterator for a RefVector


$Id: RefVectorIterator.h,v 1.10 2005/12/15 23:06:29 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/Ref.h"

namespace edm {

  template <typename C, typename T = typename Ref<C>::value_type>
  class RefVectorIterator : public std::iterator <std::random_access_iterator_tag, T > {
  public:
    typedef T value_type;
    typedef RefItem::size_type size_type;

    typedef RefVectorIterator<C, T> iterator;
    typedef std::ptrdiff_t difference;
    typedef std::vector<RefItem>::const_iterator itemIter;
    RefVectorIterator() : product_(), iter_() {}
    explicit RefVectorIterator(RefCore const& product, itemIter const& it) :
      product_(product), iter_(it) {}
    Ref<C, T> operator*() const {
      RefItem const& item = *iter_;
      getPtr<C, T>(product_, item);
      return Ref<C, T>(product_, item);
    }
    Ref<C, T> operator[](difference n) const {
      RefItem const& item = iter_[n];
      getPtr<C, T>(product_, item);
      return Ref<C, T>(product_, item);
    }
    std::auto_ptr<Ref<C, T> > operator->() const {
      RefItem const& item = *iter_;
      getPtr<C, T>(product_, item);
      return std::auto_ptr<Ref<C, T> >(new T(product_, item));
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

  template <typename C, typename T>
  inline
  RefVectorIterator<C, T> operator+(typename RefVectorIterator<C, T>::difference n, RefVectorIterator<C, T> const& iter) {
    return iter + n;
  } 
}
#endif
