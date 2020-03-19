#ifndef DataFormats_Common_RefHolder__h
#define DataFormats_Common_RefHolder__h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

#include "DataFormats/Common/interface/RefHolderBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/OffsetToBase.h"
#include <memory>
#include <typeinfo>

namespace edm {
  namespace reftobase {
    //------------------------------------------------------------------
    // Class template RefHolder<REF>
    //------------------------------------------------------------------

    template <class REF>
    class RefHolder : public RefHolderBase {
    public:
      RefHolder();
      explicit RefHolder(REF const& ref);
      void swap(RefHolder& other);
      ~RefHolder() override;
      RefHolderBase* clone() const override;

      ProductID id() const override;
      size_t key() const override;
      bool isEqualTo(RefHolderBase const& rhs) const override;
      bool fillRefIfMyTypeMatches(RefHolderBase& fillme, std::string& msg) const override;
      REF const& getRef() const;
      void setRef(REF const& r);
      std::unique_ptr<RefVectorHolderBase> makeVectorHolder() const override;
      EDProductGetter const* productGetter() const override;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      bool isAvailable() const override { return ref_.isAvailable(); }

      bool isTransient() const override { return ref_.isTransient(); }

      //Needed for ROOT storage
      CMS_CLASS_VERSION(10)
    private:
      void const* pointerToType(std::type_info const& iToType) const override;
      REF ref_;
    };

    //------------------------------------------------------------------
    // Implementation of RefHolder<REF>
    //------------------------------------------------------------------

    template <class REF>
    RefHolder<REF>::RefHolder() : RefHolderBase(), ref_() {}

    template <class REF>
    RefHolder<REF>::RefHolder(REF const& ref) : RefHolderBase(), ref_(ref) {}

    template <class REF>
    RefHolder<REF>::~RefHolder() {}

    template <class REF>
    RefHolderBase* RefHolder<REF>::clone() const {
      return new RefHolder(ref_);
    }

    template <class REF>
    ProductID RefHolder<REF>::id() const {
      return ref_.id();
    }

    template <class REF>
    bool RefHolder<REF>::isEqualTo(RefHolderBase const& rhs) const {
      RefHolder const* h(dynamic_cast<RefHolder const*>(&rhs));
      return h && (getRef() == h->getRef());
    }

    template <class REF>
    bool RefHolder<REF>::fillRefIfMyTypeMatches(RefHolderBase& fillme, std::string& msg) const {
      RefHolder* h = dynamic_cast<RefHolder*>(&fillme);
      bool conversion_worked = (h != nullptr);
      if (conversion_worked)
        h->setRef(ref_);
      else
        msg = typeid(REF).name();
      return conversion_worked;
    }

    template <class REF>
    inline REF const& RefHolder<REF>::getRef() const {
      return ref_;
    }

    template <class REF>
    EDProductGetter const* RefHolder<REF>::productGetter() const {
      return ref_.productGetter();
    }

    template <class REF>
    inline void RefHolder<REF>::swap(RefHolder& other) {
      std::swap(ref_, other.ref_);
    }

    template <class REF>
    inline void RefHolder<REF>::setRef(REF const& r) {
      ref_ = r;
    }

    template <class REF>
    void const* RefHolder<REF>::pointerToType(std::type_info const& iToType) const {
      typedef typename REF::value_type contained_type;
      if (iToType == typeid(contained_type)) {
        return ref_.get();
      }
      return pointerToBase(iToType, ref_.get());
    }
  }  // namespace reftobase
}  // namespace edm

#endif
