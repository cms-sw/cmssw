#ifndef DataFormats_Common_RefVectorHolderBase_h
#define DataFormats_Common_RefVectorHolderBase_h

#include "DataFormats/Common/interface/RefHolderBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

#include <memory>

namespace edm {
  template<typename T> class RefToBase;
  namespace reftobase {
    class RefVectorHolderBase {
    public:
      virtual ~RefVectorHolderBase() {}
      typedef size_t size_type;
      typedef RefHolderBase value_type;
      void swap(RefVectorHolderBase&) {} // nothing to swap
      virtual bool empty() const = 0;
      virtual size_type size() const = 0;
      virtual void clear() = 0;
      virtual void reserve(size_type n) = 0;
      virtual ProductID id() const = 0;
      virtual EDProductGetter const* productGetter() const = 0;
      virtual RefVectorHolderBase* clone() const = 0;
      virtual RefVectorHolderBase* cloneEmpty() const = 0;
      virtual void push_back(RefHolderBase const* r) = 0;
      // the following structure is public
      // to allow dictionary to compile
      //    protected:
      struct const_iterator_imp {
        typedef ptrdiff_t difference_type;
        const_iterator_imp() { }
        virtual ~const_iterator_imp() { }
        virtual const_iterator_imp* clone() const = 0;
        virtual void increase() = 0;
        virtual void decrease() = 0;
        virtual void increase(difference_type d) = 0;
        virtual void decrease(difference_type d) = 0;
        virtual bool equal_to(const_iterator_imp const*) const = 0;
        virtual bool less_than(const_iterator_imp const*) const = 0;
        virtual void assign(const_iterator_imp const*) = 0;
        virtual std::shared_ptr<RefHolderBase> deref() const = 0;
        virtual difference_type difference(const_iterator_imp const*) const = 0;
      };

      struct const_iterator : public std::iterator <std::random_access_iterator_tag, void*>{
        typedef std::shared_ptr<RefHolderBase> value_type;
        typedef std::ptrdiff_t difference_type;
        const_iterator() : i(0) { }
        const_iterator(const_iterator_imp* it) : i(it) { }
        const_iterator(const_iterator const& it) : i(it.isValid() ? it.i->clone() : 0) { }
        ~const_iterator() { delete i; }
        const_iterator& operator=(const_iterator const& it) {
          if(isInvalid()) i = it.i;
          else i->assign(it.i);
          return *this;
        }
        const_iterator& operator++() {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to increment an inavlid RefToBaseVector<T>::const_iterator\n");
          i->increase();
          return *this;
        }
        const_iterator operator++(int) {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to postincrement an inavlid RefToBaseVector<T>::const_iterator\n");
          const_iterator ci = *this;
          i->increase();
          return ci;
        }
        const_iterator& operator--() {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to decrement an inavlid RefToBaseVector<T>::const_iterator\n");
          i->decrease();
          return *this;
        }
        const_iterator operator--(int) {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to postdecrement an inavlid RefToBaseVector<T>::const_iterator\n");
          const_iterator ci = *this;
          i->decrease();
          return ci;
        }
        difference_type operator-(const_iterator const& o) const {
          if(isInvalid() && o.isInvalid()) return 0;
          if(isInvalid() || o.isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to compute difference with an inavlid RefToBaseVector<T>::const_iterator\n");
          return i->difference(o.i);
        }
        const_iterator operator+(difference_type n) const {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to compute sum with an inavlid RefToBaseVector<T>::const_iterator\n");
          const_iterator_imp* ii = i->clone();
          ii->increase(n);
          return const_iterator(ii);
        }
        const_iterator operator-(difference_type n) const {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to compute difference with an inavlid RefToBaseVector<T>::const_iterator\n");
          const_iterator_imp* ii = i->clone();
          ii->decrease(n);
          return const_iterator(ii);
        }
        bool operator<(const_iterator const& o) const {
          if(isInvalid() && o.isInvalid()) return false;
          if(isInvalid() || o.isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to compute < operator with an inavlid RefToBaseVector<T>::const_iterator\n");
          return i->less_than(o.i);
        }
        bool operator==(const const_iterator& ci) const {
          if(isInvalid() && ci.isInvalid()) return true;
          if(isInvalid() || ci.isInvalid()) return false;
          return i->equal_to(ci.i);
        }
        bool operator!=(const const_iterator& ci) const {
          if(isInvalid() && ci.isInvalid()) return false;
          if(isInvalid() || ci.isInvalid()) return true;
          return ! i->equal_to(ci.i);
        }
        value_type operator*() const {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to dereference an inavlid RefToBaseVector<T>::const_iterator\n");
          return i->deref();
        }
        const_iterator& operator-=(difference_type d) {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to decrement an inavlid RefToBaseVector<T>::const_iterator\n");
          i->decrease(d);
          return *this;
        }
        const_iterator& operator+=(difference_type d) {
          if(isInvalid())
            Exception::throwThis(errors::InvalidReference,
              "Trying to increment an inavlid RefToBaseVector<T>::const_iterator\n");
          i->increase(d);
          return *this;
        }
        bool isValid() const { return i != 0; }
        bool isInvalid() const { return i == 0; }

      private:
        const_iterator_imp* i;
      };

      virtual const_iterator begin() const = 0;
      virtual const_iterator end() const = 0;
      template<typename T> RefToBase<T> getRef(size_t idx) const;
      virtual void const* product() const = 0;
      virtual void reallyFillView(void const*, ProductID const&, std::vector<void const*>&) = 0;
      virtual size_t keyForIndex(size_t idx) const = 0;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const = 0;

    private:
      virtual std::shared_ptr<reftobase::RefHolderBase> refBase(size_t idx) const = 0;
    };

    template<typename T>
    RefToBase<T> RefVectorHolderBase::getRef(size_t idx) const {
      std::shared_ptr<reftobase::RefHolderBase> rb = refBase(idx);
      return RefToBase<T>(rb);
    }

    // Free swap function
    inline
    void swap(RefVectorHolderBase& lhs, RefVectorHolderBase& rhs) {
      lhs.swap(rhs);
    }
  }
}

#endif
