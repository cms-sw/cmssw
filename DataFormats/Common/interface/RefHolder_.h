#ifndef DataFormats_Common_RefHolder__h
#define DataFormats_Common_RefHolder__h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

#include "DataFormats/Common/interface/RefHolderBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/OffsetToBase.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
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
      virtual ~RefHolder();
      virtual RefHolderBase* clone() const GCC11_OVERRIDE;

      virtual ProductID id() const GCC11_OVERRIDE;
      virtual size_t key() const GCC11_OVERRIDE;
      virtual bool isEqualTo(RefHolderBase const& rhs) const GCC11_OVERRIDE;
      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const GCC11_OVERRIDE;
      REF const& getRef() const;
      void setRef(REF const& r);
      virtual std::auto_ptr<RefVectorHolderBase> makeVectorHolder() const GCC11_OVERRIDE;
      virtual EDProductGetter const* productGetter() const GCC11_OVERRIDE;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const GCC11_OVERRIDE { return ref_.isAvailable(); }

      virtual bool isTransient() const GCC11_OVERRIDE { return ref_.isTransient(); }

      //Needed for ROOT storage
      CMS_CLASS_VERSION(10)
    private:
      virtual void const* pointerToType(std::type_info const& iToType) const GCC11_OVERRIDE;
      REF ref_;
    };

    //------------------------------------------------------------------
    // Implementation of RefHolder<REF>
    //------------------------------------------------------------------

    template <class REF>
    RefHolder<REF>::RefHolder() : 
      RefHolderBase(), ref_()
    { }
  
    template <class REF>
    RefHolder<REF>::RefHolder(REF const& ref) : 
      RefHolderBase(), ref_(ref) 
    { }

    template <class REF>
    RefHolder<REF>::~RefHolder() 
    { }

    template <class REF>
    RefHolderBase* 
    RefHolder<REF>::clone() const
    {
      return new RefHolder(ref_);
    }

    template <class REF>
    ProductID
    RefHolder<REF>::id() const 
    {
      return ref_.id();
    }

    template <class REF>
    bool
    RefHolder<REF>::isEqualTo(RefHolderBase const& rhs) const 
    { 
      RefHolder const* h(dynamic_cast<RefHolder const*>(&rhs));
      return h && (getRef() == h->getRef());
    }

    template <class REF>
    bool 
    RefHolder<REF>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					   std::string& msg) const
    {
      RefHolder* h = dynamic_cast<RefHolder*>(&fillme);
      bool conversion_worked = (h != 0);
      if (conversion_worked)
	h->setRef(ref_);
      else
	msg = typeid(REF).name();
      return conversion_worked;
    }

    template <class REF>
    inline
    REF const&
    RefHolder<REF>::getRef() const
    {
      return ref_;
    }

    template<class REF>
    EDProductGetter const* RefHolder<REF>::productGetter() const {
      return ref_.productGetter();
    }

    template <class REF>
    inline
    void
    RefHolder<REF>::swap(RefHolder& other)
    {
      std::swap(ref_, other.ref_);
    }

    template <class REF>
    inline
    void
    RefHolder<REF>::setRef(REF const& r)
    {
      ref_ = r;
    }

    template <class REF>
    void const* 
    RefHolder<REF>::pointerToType(std::type_info const& iToType) const {
      typedef typename REF::value_type contained_type;
      if(iToType == typeid(contained_type)) {
        return ref_.get();
      }
      return pointerToBase(iToType, ref_.get());
    }
  } // namespace reftobase
}

#endif
