#ifndef DataFormats_Common_VectorHolder_h
#define DataFormats_Common_VectorHolder_h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/BaseVectorHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include <memory>

namespace edm {
  namespace reftobase {

    class RefVectorHolderBase;

    template <class T, class REFV>
    class VectorHolder : public BaseVectorHolder<T> {
    public:
      typedef BaseVectorHolder<T> base_type;
      typedef typename base_type::size_type size_type;
      typedef typename base_type::element_type element_type;
      typedef typename base_type::base_ref_type base_ref_type;
      typedef typename base_type::const_iterator const_iterator;
      typedef REFV ref_vector_type;

      VectorHolder() : base_type() {}
      VectorHolder(VectorHolder const& rh) : base_type(rh), refVector_(rh.refVector_) {}
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
      VectorHolder(VectorHolder&& rh) noexcept : base_type(std::forward(rh)), refVector_(std::move(rh.refVector_)) {}
#endif

      explicit VectorHolder(const ref_vector_type& iRefVector) : base_type(), refVector_(iRefVector) {}
      explicit VectorHolder(const ProductID& iId) : base_type(), refVector_(iId) {}
      ~VectorHolder() noexcept override {}
      base_type* clone() const override { return new VectorHolder(*this); }
      base_type* cloneEmpty() const override { return new VectorHolder(refVector_.id()); }
      base_ref_type const at(size_type idx) const override { return base_ref_type(refVector_.at(idx)); }
      bool empty() const override { return refVector_.empty(); }
      size_type size() const override { return refVector_.size(); }
      //size_type capacity() const { return refVector_.capacity(); }
      //void reserve(size_type n) { refVector_.reserve(n); }
      void clear() override { refVector_.clear(); }
      ProductID id() const override { return refVector_.id(); }
      EDProductGetter const* productGetter() const override { return refVector_.productGetter(); }
      void swap(VectorHolder& other) noexcept {
        this->BaseVectorHolder<T>::swap(other);
        refVector_.swap(other.refVector_);
      }
      VectorHolder& operator=(VectorHolder const& rhs) {
        VectorHolder temp(rhs);
        this->swap(temp);
        return *this;
      }
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
      VectorHolder& operator=(VectorHolder&& rhs) noexcept {
        base_type::operator=(std::forward(rhs));
        refVector_ = std::move(rhs.refVector_);
        return *this;
      }
#endif

      const_iterator begin() const override {
        return const_iterator(new const_iterator_imp_specific(refVector_.begin()));
      }
      const_iterator end() const override { return const_iterator(new const_iterator_imp_specific(refVector_.end())); }
      void push_back(const BaseHolder<T>* r) override {
        typedef Holder<T, typename REFV::value_type> holder_type;
        const holder_type* h = dynamic_cast<const holder_type*>(r);
        if (h == nullptr)
          Exception::throwThis(errors::InvalidReference,
                               "In VectorHolder<T, REFV> trying to push_back wrong reference type");
        refVector_.push_back(h->getRef());
      }
      std::unique_ptr<RefVectorHolderBase> vectorHolder() const override {
        return std::unique_ptr<RefVectorHolderBase>(new RefVectorHolder<REFV>(refVector_));
      }

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      bool isAvailable() const override { return refVector_.isAvailable(); }

      //Used by ROOT storage
      CMS_CLASS_VERSION(10)

    private:
      typedef typename base_type::const_iterator_imp const_iterator_imp;

      ref_vector_type refVector_;

      // the following structure is public
      // to allow dictionary to compile
    public:
      struct const_iterator_imp_specific : public const_iterator_imp {
        typedef ptrdiff_t difference_type;
        const_iterator_imp_specific() {}
        explicit const_iterator_imp_specific(const typename REFV::const_iterator& it) : i(it) {}
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
        const typename ref_vector_type::const_iterator& dc(const const_iterator_imp* o) const {
          if (o == nullptr)
            Exception::throwThis(errors::InvalidReference,
                                 "In RefToBaseVector<T> trying to dereference a null pointer");
          const const_iterator_imp_specific* oo = dynamic_cast<const const_iterator_imp_specific*>(o);
          if (oo == nullptr)
            Exception::throwThis(errors::InvalidReference,
                                 "In RefToBaseVector<T> trying to cast iterator to wrong type ");
          return oo->i;
        }
        typename ref_vector_type::const_iterator i;
      };
    };

    // Free swap function
    template <typename T, typename REFV>
    inline void swap(VectorHolder<T, REFV>& lhs, VectorHolder<T, REFV>& rhs) noexcept {
      lhs.swap(rhs);
    }
  }  // namespace reftobase
}  // namespace edm

#endif
