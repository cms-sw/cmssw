#ifndef DataFormats_Common_RefHolder__h
#define DataFormats_Common_RefHolder__h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

#include "DataFormats/Common/interface/RefHolderBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "Reflex/Object.h"
#include "Reflex/Type.h"
#include <memory>

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
      virtual RefHolderBase* clone() const;

      virtual ProductID id() const;
      virtual size_t key() const;
      virtual bool isEqualTo(RefHolderBase const& rhs) const;
      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;
      REF const& getRef() const;
      void setRef(REF const& r);
      virtual std::auto_ptr<RefVectorHolderBase> makeVectorHolder() const;
      virtual EDProductGetter const* productGetter() const;
      virtual bool hasProductCache() const;
      virtual void const * product() const;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const { return ref_.isAvailable(); }

      //Needed for ROOT storage
      CMS_CLASS_VERSION(10)
    private:
      virtual void const* pointerToType(Reflex::Type const& iToType) const;
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

    template<class REF>
    bool RefHolder<REF>::hasProductCache() const {
      return ref_.hasProductCache();
    }

    template<class REF>
    void const * RefHolder<REF>::product() const {
      return ref_.product();
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
    RefHolder<REF>::pointerToType(Reflex::Type const& iToType) const 
    {
      typedef typename REF::value_type contained_type;
      static const Reflex::Type s_type(Reflex::Type::ByTypeInfo(typeid(contained_type)));
    
      // The const_cast below is needed because
      // Object's constructor requires a pointer to
      // non-const void, although the implementation does not, of
      // course, modify the object to which the pointer points.
      Reflex::Object obj(s_type, const_cast<void*>(static_cast<const void*>(ref_.get())));
      if ( s_type == iToType ) return obj.Address();
      Reflex::Object cast = obj.CastObject(iToType);
      return cast.Address(); // returns void*, after pointer adjustment
    }
  } // namespace reftobase
}

#endif
