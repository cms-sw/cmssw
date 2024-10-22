#ifndef DataFormats_Common_IndirectVectorHolder_h
#define DataFormats_Common_IndirectVectorHolder_h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/BaseVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"
#include "DataFormats/Common/interface/IndirectHolder.h"
#include <memory>

namespace edm {
  namespace reftobase {

    template <typename T>
    class IndirectVectorHolder : public BaseVectorHolder<T> {
    public:
      typedef BaseVectorHolder<T> base_type;
      typedef typename base_type::size_type size_type;
      typedef typename base_type::element_type element_type;
      typedef typename base_type::base_ref_type base_ref_type;
      typedef typename base_type::const_iterator const_iterator;

      IndirectVectorHolder();
      IndirectVectorHolder(const IndirectVectorHolder& other);
      IndirectVectorHolder(std::shared_ptr<RefVectorHolderBase> p);
      IndirectVectorHolder(RefVectorHolderBase* p);
      ~IndirectVectorHolder() override;
      IndirectVectorHolder& operator=(IndirectVectorHolder const& rhs);
      void swap(IndirectVectorHolder& other);
      BaseVectorHolder<T>* clone() const override;
      BaseVectorHolder<T>* cloneEmpty() const override;
      ProductID id() const override;
      EDProductGetter const* productGetter() const override;
      bool empty() const override;
      size_type size() const override;
      void clear() override;
      base_ref_type const at(size_type idx) const override;
      std::unique_ptr<reftobase::RefVectorHolderBase> vectorHolder() const override {
        return std::unique_ptr<reftobase::RefVectorHolderBase>(helper_->clone());
      }
      void push_back(const BaseHolder<T>* r) override {
        typedef IndirectHolder<T> holder_type;
        const holder_type* h = dynamic_cast<const holder_type*>(r);
        if (h == nullptr)
          Exception::throwThis(errors::InvalidReference,
                               "In IndirectHolder<T> trying to push_back wrong reference type");
        helper_->push_back(h->helper_);
      }

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      bool isAvailable() const override { return helper_->isAvailable(); }

      //Used by ROOT storage
      CMS_CLASS_VERSION(10)

    private:
      typedef typename base_type::const_iterator_imp const_iterator_imp;
      RefVectorHolderBase* helper_;

    public:
      struct const_iterator_imp_specific : public const_iterator_imp {
        typedef ptrdiff_t difference_type;
        const_iterator_imp_specific() {}
        explicit const_iterator_imp_specific(const typename RefVectorHolderBase::const_iterator& it) : i(it) {}
        ~const_iterator_imp_specific() override {}
        const_iterator_imp_specific* clone() const override { return new const_iterator_imp_specific(i); }
        void increase() override { ++i; }
        void decrease() override { --i; }
        void increase(difference_type d) override { i += d; }
        void decrease(difference_type d) override { i -= d; }
        bool equal_to(const const_iterator_imp* o) const override { return i == dc(o); }
        bool less_than(const const_iterator_imp* o) const override { return i < dc(o); }
        void assign(const const_iterator_imp* o) override { i = dc(o); }
        base_ref_type deref() const override { return base_ref_type(*i); }
        difference_type difference(const const_iterator_imp* o) const override { return i - dc(o); }

      private:
        const typename RefVectorHolderBase::const_iterator& dc(const const_iterator_imp* o) const {
          if (o == nullptr) {
            Exception::throwThis(edm::errors::InvalidReference,
                                 "In IndirectVectorHolder trying to dereference a null pointer");
          }
          const const_iterator_imp_specific* oo = dynamic_cast<const const_iterator_imp_specific*>(o);
          if (oo == nullptr) {
            Exception::throwThis(errors::InvalidReference,
                                 "In IndirectVectorHolder trying to cast iterator to wrong type ");
          }
          return oo->i;
        }
        typename RefVectorHolderBase::const_iterator i;
      };

      const_iterator begin() const override {
        return const_iterator(new const_iterator_imp_specific(helper_->begin()));
      }
      const_iterator end() const override { return const_iterator(new const_iterator_imp_specific(helper_->end())); }
    };

    template <typename T>
    IndirectVectorHolder<T>::IndirectVectorHolder() : BaseVectorHolder<T>(), helper_(nullptr) {}

    template <typename T>
    IndirectVectorHolder<T>::IndirectVectorHolder(std::shared_ptr<RefVectorHolderBase> p)
        : BaseVectorHolder<T>(), helper_(p->clone()) {}

    template <typename T>
    IndirectVectorHolder<T>::IndirectVectorHolder(RefVectorHolderBase* p) : BaseVectorHolder<T>(), helper_(p) {}

    template <typename T>
    IndirectVectorHolder<T>::IndirectVectorHolder(const IndirectVectorHolder& other)
        : BaseVectorHolder<T>(), helper_(other.helper_->clone()) {}

    template <typename T>
    IndirectVectorHolder<T>::~IndirectVectorHolder() {
      delete helper_;
    }

    template <typename T>
    inline void IndirectVectorHolder<T>::swap(IndirectVectorHolder& other) {
      this->BaseVectorHolder<T>::swap(other);
      std::swap(helper_, other.helper_);
    }

    template <typename T>
    inline IndirectVectorHolder<T>& IndirectVectorHolder<T>::operator=(IndirectVectorHolder const& rhs) {
      IndirectVectorHolder temp(rhs);
      swap(temp);
      return *this;
    }

    template <typename T>
    BaseVectorHolder<T>* IndirectVectorHolder<T>::clone() const {
      return new IndirectVectorHolder<T>(*this);
    }

    template <typename T>
    BaseVectorHolder<T>* IndirectVectorHolder<T>::cloneEmpty() const {
      return new IndirectVectorHolder<T>(helper_->cloneEmpty());
    }

    template <typename T>
    ProductID IndirectVectorHolder<T>::id() const {
      return helper_->id();
    }

    template <typename T>
    EDProductGetter const* IndirectVectorHolder<T>::productGetter() const {
      return helper_->productGetter();
    }

    template <typename T>
    bool IndirectVectorHolder<T>::empty() const {
      return helper_->empty();
    }

    template <typename T>
    typename IndirectVectorHolder<T>::size_type IndirectVectorHolder<T>::size() const {
      return helper_->size();
    }

    template <typename T>
    void IndirectVectorHolder<T>::clear() {
      return helper_->clear();
    }

    template <typename T>
    typename IndirectVectorHolder<T>::base_ref_type const IndirectVectorHolder<T>::at(size_type idx) const {
      return helper_ ? helper_->template getRef<T>(idx) : typename IndirectVectorHolder<T>::base_ref_type();
    }

    // Free swap function
    template <typename T>
    inline void swap(IndirectVectorHolder<T>& lhs, IndirectVectorHolder<T>& rhs) {
      lhs.swap(rhs);
    }
  }  // namespace reftobase
}  // namespace edm

#endif
