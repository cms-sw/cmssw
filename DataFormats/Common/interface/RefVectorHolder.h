#ifndef DataFormats_Common_RefVectorHolder_h
#define DataFormats_Common_RefVectorHolder_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  namespace reftobase {
    class RefHolderBase;
    template <typename REF>
    class RefHolder;

    template <typename REFV>
    class RefVectorHolder : public RefVectorHolderBase {
    public:
      RefVectorHolder() : RefVectorHolderBase() {}
      RefVectorHolder(REFV const& refs) : RefVectorHolderBase(), refs_(refs) {}
      explicit RefVectorHolder(ProductID const& iId) : RefVectorHolderBase(), refs_(iId) {}
      ~RefVectorHolder() override {}
      void swap(RefVectorHolder& other);
      RefVectorHolder& operator=(RefVectorHolder const& rhs);
      bool empty() const override;
      size_type size() const override;
      void clear() override;
      void push_back(RefHolderBase const* r) override;
      void reserve(size_type n) override;
      ProductID id() const override;
      EDProductGetter const* productGetter() const override;
      RefVectorHolder<REFV>* clone() const override;
      RefVectorHolder<REFV>* cloneEmpty() const override;
      void setRefs(REFV const& refs);
      size_t keyForIndex(size_t idx) const override;

      //Needed for ROOT storage
      CMS_CLASS_VERSION(10)

    private:
      typedef typename RefVectorHolderBase::const_iterator_imp const_iterator_imp;

    public:
      struct const_iterator_imp_specific : public const_iterator_imp {
        typedef ptrdiff_t difference_type;
        const_iterator_imp_specific() {}
        explicit const_iterator_imp_specific(typename REFV::const_iterator const& it) : i(it) {}
        ~const_iterator_imp_specific() override {}
        const_iterator_imp_specific* clone() const override { return new const_iterator_imp_specific(i); }
        void increase() override { ++i; }
        void decrease() override { --i; }
        void increase(difference_type d) override { i += d; }
        void decrease(difference_type d) override { i -= d; }
        bool equal_to(const_iterator_imp const* o) const override { return i == dc(o); }
        bool less_than(const_iterator_imp const* o) const override { return i < dc(o); }
        void assign(const_iterator_imp const* o) override { i = dc(o); }
        std::shared_ptr<RefHolderBase> deref() const override;
        difference_type difference(const_iterator_imp const* o) const override { return i - dc(o); }

      private:
        typename REFV::const_iterator const& dc(const_iterator_imp const* o) const {
          if (o == nullptr) {
            Exception::throwThis(errors::InvalidReference, "In RefVectorHolder trying to dereference a null pointer\n");
          }
          const_iterator_imp_specific const* oo = dynamic_cast<const_iterator_imp_specific const*>(o);
          if (oo == nullptr) {
            Exception::throwThis(errors::InvalidReference,
                                 "In RefVectorHolder trying to cast iterator to wrong type\n");
          }
          return oo->i;
        }
        typename REFV::const_iterator i;
      };

      typedef typename RefVectorHolderBase::const_iterator const_iterator;

      const_iterator begin() const override { return const_iterator(new const_iterator_imp_specific(refs_.begin())); }
      const_iterator end() const override { return const_iterator(new const_iterator_imp_specific(refs_.end())); }

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      bool isAvailable() const override { return refs_.isAvailable(); }

    private:
      std::shared_ptr<reftobase::RefHolderBase> refBase(size_t idx) const override;
      REFV refs_;
    };

    //
    // implementations for RefVectorHolder<REFV>
    //

    template <typename REFV>
    inline void RefVectorHolder<REFV>::swap(RefVectorHolder<REFV>& other) {
      this->RefVectorHolderBase::swap(other);
      refs_.swap(other.refs_);
    }

    template <typename REFV>
    inline RefVectorHolder<REFV>& RefVectorHolder<REFV>::operator=(RefVectorHolder<REFV> const& rhs) {
      RefVectorHolder<REFV> temp(rhs);
      this->swap(temp);
      return *this;
    }

    template <typename REFV>
    inline bool RefVectorHolder<REFV>::empty() const {
      return refs_.empty();
    }

    template <typename REFV>
    inline typename RefVectorHolder<REFV>::size_type RefVectorHolder<REFV>::size() const {
      return refs_.size();
    }

    template <typename REFV>
    inline void RefVectorHolder<REFV>::clear() {
      return refs_.clear();
    }

    template <typename REFV>
    inline void RefVectorHolder<REFV>::reserve(size_type n) {
      typename REFV::size_type s = n;
      refs_.reserve(s);
    }

    template <typename REFV>
    inline ProductID RefVectorHolder<REFV>::id() const {
      return refs_.id();
    }

    template <typename REFV>
    inline EDProductGetter const* RefVectorHolder<REFV>::productGetter() const {
      return refs_.productGetter();
    }

    template <typename REFV>
    inline RefVectorHolder<REFV>* RefVectorHolder<REFV>::clone() const {
      return new RefVectorHolder<REFV>(*this);
    }

    template <typename REFV>
    inline RefVectorHolder<REFV>* RefVectorHolder<REFV>::cloneEmpty() const {
      return new RefVectorHolder<REFV>(id());
    }

    template <typename REFV>
    inline void RefVectorHolder<REFV>::setRefs(REFV const& refs) {
      refs_ = refs;
    }

    template <typename REFV>
    inline size_t RefVectorHolder<REFV>::keyForIndex(size_t idx) const {
      return refs_[idx].key();
    }

    // Free swap function
    template <typename REFV>
    inline void swap(RefVectorHolder<REFV>& lhs, RefVectorHolder<REFV>& rhs) {
      lhs.swap(rhs);
    }
  }  // namespace reftobase
}  // namespace edm

#include "DataFormats/Common/interface/RefHolder.h"

namespace edm {
  namespace reftobase {

    template <typename REFV>
    void RefVectorHolder<REFV>::push_back(RefHolderBase const* h) {
      typedef typename REFV::value_type REF;
      RefHolder<REF> const* rh = dynamic_cast<RefHolder<REF> const*>(h);
      if (rh == nullptr) {
        Exception::throwThis(errors::InvalidReference,
                             "RefVectorHolder: attempting to cast a RefHolderBase "
                             "to an invalid type.\nExpected: ",
                             typeid(REF).name(),
                             "\n");
      }
      refs_.push_back(rh->getRef());
    }

    template <typename REFV>
    std::shared_ptr<RefHolderBase> RefVectorHolder<REFV>::refBase(size_t idx) const {
      return std::shared_ptr<RefHolderBase>(std::make_shared<RefHolder<typename REFV::value_type> >(refs_[idx]));
    }

    template <typename REFV>
    std::shared_ptr<RefHolderBase> RefVectorHolder<REFV>::const_iterator_imp_specific::deref() const {
      return std::shared_ptr<RefHolderBase>(std::make_shared<RefHolder<typename REFV::value_type> >(*i));
    }
  }  // namespace reftobase
}  // namespace edm

#endif
