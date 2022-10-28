#ifndef DataFormats_Common_IndirectHolder_h
#define DataFormats_Common_IndirectHolder_h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/BaseHolder.h"
#include "DataFormats/Common/interface/RefHolderBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <memory>

namespace edm {
  template <typename T>
  class RefToBase;

  namespace reftobase {

    template <typename T>
    class IndirectVectorHolder;
    class RefVectorHolderBase;
    class RefHolderBase;

    //------------------------------------------------------------------
    // Class template IndirectHolder<T>
    //------------------------------------------------------------------

    template <typename T>
    class IndirectHolder : public BaseHolder<T> {
    public:
      // It may be better to use unique_ptr<RefHolderBase> in
      // this constructor, so that the cloning can be avoided. I'm not
      // sure if use of unique_ptr here causes any troubles elsewhere.
      IndirectHolder() : BaseHolder<T>(), helper_(nullptr) {}
      IndirectHolder(std::shared_ptr<RefHolderBase> p);
      template <typename U>
      IndirectHolder(std::unique_ptr<U> p) : helper_(p.release()) {}
      IndirectHolder(IndirectHolder const& other);
      IndirectHolder& operator=(IndirectHolder const& rhs);
      void swap(IndirectHolder& other);
      ~IndirectHolder() override;

      BaseHolder<T>* clone() const override;
      T const* getPtr() const override;
      ProductID id() const override;
      size_t key() const override;
      bool isEqualTo(BaseHolder<T> const& rhs) const override;

      std::unique_ptr<RefHolderBase> holder() const override;
      std::unique_ptr<BaseVectorHolder<T> > makeVectorHolder() const override;
      EDProductGetter const* productGetter() const override;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      bool isAvailable() const override { return helper_->isAvailable(); }

      bool isTransient() const override { return helper_->isTransient(); }

      //Used by ROOT storage
      CMS_CLASS_VERSION(10)

    private:
      friend class RefToBase<T>;
      friend class IndirectVectorHolder<T>;
      RefHolderBase* helper_;
    };
    //------------------------------------------------------------------
    // Implementation of IndirectHolder<T>
    //------------------------------------------------------------------
    template <typename T>
    inline IndirectHolder<T>::IndirectHolder(std::shared_ptr<RefHolderBase> p) : BaseHolder<T>(), helper_(p->clone()) {}

    template <typename T>
    inline IndirectHolder<T>::IndirectHolder(IndirectHolder const& other)
        : BaseHolder<T>(other), helper_(other.helper_->clone()) {}

    template <typename T>
    inline void IndirectHolder<T>::swap(IndirectHolder& other) {
      this->BaseHolder<T>::swap(other);
      std::swap(helper_, other.helper_);
    }

    template <typename T>
    inline IndirectHolder<T>& IndirectHolder<T>::operator=(IndirectHolder const& rhs) {
      IndirectHolder temp(rhs);
      swap(temp);
      return *this;
    }

    template <typename T>
    IndirectHolder<T>::~IndirectHolder() {
      delete helper_;
    }

    template <typename T>
    BaseHolder<T>* IndirectHolder<T>::clone() const {
      return new IndirectHolder<T>(*this);
    }

    template <typename T>
    T const* IndirectHolder<T>::getPtr() const {
      return helper_->template getPtr<T>();
    }

    template <typename T>
    ProductID IndirectHolder<T>::id() const {
      return helper_->id();
    }

    template <typename T>
    size_t IndirectHolder<T>::key() const {
      return helper_->key();
    }

    template <typename T>
    inline EDProductGetter const* IndirectHolder<T>::productGetter() const {
      return helper_->productGetter();
    }

    template <typename T>
    bool IndirectHolder<T>::isEqualTo(BaseHolder<T> const& rhs) const {
      IndirectHolder const* h = dynamic_cast<IndirectHolder const*>(&rhs);
      return h && helper_->isEqualTo(*h->helper_);
    }

    template <typename T>
    std::unique_ptr<RefHolderBase> IndirectHolder<T>::holder() const {
      return std::unique_ptr<RefHolderBase>(helper_->clone());
    }

    // Free swap function
    template <typename T>
    inline void swap(IndirectHolder<T>& lhs, IndirectHolder<T>& rhs) {
      lhs.swap(rhs);
    }
  }  // namespace reftobase

}  // namespace edm

#include "DataFormats/Common/interface/IndirectVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"

namespace edm {
  namespace reftobase {
    template <typename T>
    std::unique_ptr<BaseVectorHolder<T> > IndirectHolder<T>::makeVectorHolder() const {
      std::unique_ptr<RefVectorHolderBase> p = helper_->makeVectorHolder();
      std::shared_ptr<RefVectorHolderBase> sp(p.release());
      return std::unique_ptr<BaseVectorHolder<T> >(new IndirectVectorHolder<T>(sp));
    }
  }  // namespace reftobase
}  // namespace edm

#endif
