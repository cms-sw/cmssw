#ifndef DataFormats_Common_BaseVectorHolder_h
#define DataFormats_Common_BaseVectorHolder_h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/BaseHolder.h"
#include <memory>

namespace edm {
  class ProductID;
  template<typename T> class RefToBase;
  namespace reftobase {
    template<typename T>
    class BaseVectorHolder {
    public:
      typedef size_t       size_type;
      typedef T            element_type;
      typedef RefToBase<T> base_ref_type;
      BaseVectorHolder() {}
      virtual ~BaseVectorHolder() {}
      virtual BaseVectorHolder* clone() const = 0;
      virtual BaseVectorHolder* cloneEmpty() const = 0;
      virtual base_ref_type const at(size_type idx) const = 0;
      virtual bool empty() const = 0;

      virtual size_type size() const = 0;
      //virtual size_type capacity() const = 0;
      //virtual void reserve(size_type n) = 0;
      virtual void clear() = 0;
      virtual ProductID id() const = 0;
      virtual EDProductGetter const* productGetter() const = 0;
      void swap(BaseVectorHolder&) {} // nothing to swap

      // the following structure is public
      // to allow reflex dictionary to compile
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
        virtual base_ref_type deref() const = 0;
        virtual difference_type difference(const_iterator_imp const*) const = 0;
      };

      struct const_iterator : public std::iterator <std::random_access_iterator_tag, RefToBase<T> >{
        typedef base_ref_type value_type;
        typedef std::auto_ptr<value_type> pointer;
        typedef std::ptrdiff_t difference_type;

        const_iterator() : i(0) { }
        const_iterator(const_iterator_imp* it) : i(it) { }
        const_iterator(const_iterator const& it) : i(it.isValid() ? it.i->clone() : 0) { }
        ~const_iterator() { delete i; }
        const_iterator& operator=(const_iterator const& it) {
          if (isInvalid()) i = it.i;
          else i->assign(it.i);
          return *this;
        }
        const_iterator& operator++() {
          throwInvalidReference(isInvalid(), "increment");
          i->increase();
          return *this;
        }
        const_iterator operator++(int) {
          throwInvalidReference(isInvalid(), "postincrement");
          const_iterator ci = *this;
          i->increase();
          return ci;
        }
        const_iterator& operator--() {
          throwInvalidReference(isInvalid(), "decrement");
          i->decrease();
          return *this;
        }
        const_iterator operator--(int) {
          throwInvalidReference(isInvalid(), "postdecrement");
          const_iterator ci = *this;
          i->decrease();
          return ci;
        }
        difference_type operator-(const_iterator const& o) const {
          if (isInvalid() && o.isInvalid()) return 0;
          throwInvalidReference(isInvalid() || o.isInvalid(), "compute difference with");
          return i->difference(o.i);
        }
        const_iterator operator+(difference_type n) const {
          throwInvalidReference(isInvalid(), "compute sum with");
          const_iterator_imp* ii = i->clone();
          ii->increase(n);
          return const_iterator(ii);
        }
        const_iterator operator-(difference_type n) const {
          throwInvalidReference(isInvalid(), "compute difference with");
          const_iterator_imp* ii = i->clone();
          ii->decrease(n);
          return const_iterator(ii);
        }
        bool operator<(const_iterator const& o) const {
          if (isInvalid() && o.isInvalid()) return false;
          throwInvalidReference(isInvalid() || o.isInvalid(), "compute < operator with");
          return i->less_than(o.i);
        }
        bool operator==(const_iterator const& ci) const {
          if (isInvalid() && ci.isInvalid()) return true;
          if (isInvalid() || ci.isInvalid()) return false;
          return i->equal_to(ci.i);
        }
        bool operator!=(const_iterator const& ci) const {
          if (isInvalid() && ci.isInvalid()) return false;
          if (isInvalid() || ci.isInvalid()) return true;
          return ! i->equal_to(ci.i);
        }
        value_type operator*() const {
          throwInvalidReference(isInvalid(), "dereference");
          return i->deref();
        }
        pointer operator->()const {
          return pointer(new value_type(operator*()));
        }
        const_iterator& operator+=(difference_type d) {
          throwInvalidReference(isInvalid(), "increment");
          i->increase(d);
          return *this;
        }
        const_iterator& operator-=(difference_type d) {
          throwInvalidReference(isInvalid(), "decrement");
          i->decrease(d);
          return *this;
        }
        bool isValid() const { return i != 0; }
        bool isInvalid() const { return i == 0; }

        void throwInvalidReference(bool isInvalid, char const* why) const {
          if (isInvalid) {
            Exception::throwThis(errors::InvalidReference, "Trying to ", why, " an invalid RefToBaseVector<T>::const_iterator");
          }
        }

      private:
        const_iterator_imp* i;
      };

      virtual const_iterator begin() const = 0;
      virtual const_iterator end() const = 0;
      virtual void push_back(BaseHolder<T> const*) = 0;
      virtual std::auto_ptr<RefVectorHolderBase> vectorHolder() const = 0;
      virtual void const* product() const = 0;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const = 0;

      //Used by ROOT storage
      CMS_CLASS_VERSION(3)
    };

    // Free swap function
    template<typename T>
    inline
    void
    swap(BaseVectorHolder<T>& lhs, BaseVectorHolder<T>& rhs) {
      lhs.swap(rhs);
    }
  }
}

#endif
