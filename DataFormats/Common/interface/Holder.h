
#ifndef DataFormats_Common_Holder_h
#define DataFormats_Common_Holder_h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/BaseHolder.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <memory>

namespace edm {
  namespace reftobase {
   //------------------------------------------------------------------
    // Class template Holder<T,REF>
    //------------------------------------------------------------------

    template <class T, class REF>
    class Holder : public BaseHolder<T> {
    public:
      Holder();
      Holder(Holder const& other);
      explicit Holder(REF const& iRef);
      Holder& operator= (Holder const& rhs);
      void swap(Holder& other);
      virtual ~Holder();
      virtual BaseHolder<T>* clone() const GCC11_OVERRIDE;

      virtual T const* getPtr() const GCC11_OVERRIDE;
      virtual ProductID id() const GCC11_OVERRIDE;
      virtual size_t key() const GCC11_OVERRIDE;
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const GCC11_OVERRIDE;
      REF const& getRef() const;

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const GCC11_OVERRIDE;

      virtual std::auto_ptr<RefHolderBase> holder() const GCC11_OVERRIDE {
	return std::auto_ptr<RefHolderBase>( new RefHolder<REF>( ref_ ) );
      }
      virtual std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() const GCC11_OVERRIDE;
      virtual EDProductGetter const* productGetter() const GCC11_OVERRIDE;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const GCC11_OVERRIDE { return ref_.isAvailable(); }

      virtual bool isTransient() const GCC11_OVERRIDE { return ref_.isTransient(); }

      //Used by ROOT storage
      CMS_CLASS_VERSION(10)

    private:
      REF ref_;
    };

    //------------------------------------------------------------------
    // Implementation of Holder<T,REF>
    //------------------------------------------------------------------

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder() : BaseHolder<T>(),
      ref_()
    {  }

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder(Holder const& other) : BaseHolder<T>(other),
      ref_(other.ref_)
    { }

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder(REF const& r) : BaseHolder<T>(),
      ref_(r)
    { }

    template <class T, class REF>
    inline
    Holder<T,REF> &
    Holder<T,REF>::operator=(Holder const& rhs)
    {
      Holder temp(rhs);
      swap(temp);
      return *this;
    }

    template <class T, class REF>
    inline
    void
    Holder<T,REF>::swap(Holder& other)
    {
      std::swap(ref_, other.ref_);
    }

    template <class T, class REF>
    inline
    Holder<T,REF>::~Holder()
    { }

    template <class T, class REF>
    inline
    BaseHolder<T>*
    Holder<T,REF>::clone() const 
    {
      return new Holder(*this);
    }

    template <class T, class REF>
    inline
    T const*
    Holder<T,REF>::getPtr() const
    {
      return ref_.operator->();
    }

    template <class T, class REF>
    inline
    ProductID
    Holder<T,REF>::id() const
    {
      return ref_.id();
    }

    template <class T, class REF>
    inline
    bool
    Holder<T,REF>::isEqualTo(BaseHolder<T> const& rhs) const
    {
      Holder const* h = dynamic_cast<Holder const*>(&rhs);
      return h && (getRef() == h->getRef());
      //       if (h == 0) return false;
      //       return getRef() == h->getRef();
    }

    template <class T, class REF>
    inline
    REF const&
    Holder<T,REF>::getRef() const
    {
      return ref_;
    }

    template <class T, class REF>
    inline
    EDProductGetter const* Holder<T,REF>::productGetter() const {
      return ref_.productGetter();
    }

    template <class T, class REF>
    bool
    Holder<T,REF>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const
    {
      RefHolder<REF>* h = dynamic_cast<RefHolder<REF>*>(&fillme);
      bool conversion_worked = (h != 0);

      if (conversion_worked)
 	h->setRef(ref_);
      else
	msg = typeid(REF).name();

      return conversion_worked;
    }

  }
}

#include "DataFormats/Common/interface/HolderToVectorTrait.h"
#include "DataFormats/Common/interface/Ref.h"

namespace edm {
  namespace reftobase {

    template <typename T, typename REF>
    std::auto_ptr<BaseVectorHolder<T> > Holder<T,REF>::makeVectorHolder() const {
      typedef typename HolderToVectorTrait<T, REF>::type helper;
      return helper::makeVectorHolder();
    }
  }
}

#include "DataFormats/Common/interface/RefKeyTrait.h"

namespace edm {
  namespace reftobase {

    template <class T, class REF>
    inline
    size_t
    Holder<T,REF>::key() const
    {
      typedef typename RefKeyTrait<REF>::type helper;
      return helper::key( ref_ );
    }
    
  }
}
#endif
