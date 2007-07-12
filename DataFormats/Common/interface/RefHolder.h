#ifndef Common_RefHolder_h
#define Common_RefHolder_h
#include "DataFormats/Common/interface/RefHolderBase.h"
#include "Reflex/Object.h"
#include "Reflex/Type.h"

namespace edm {
  namespace reftobase {
    // The following makes ROOT::Reflex::Type available as reftobase::Type,
    // etc.
    using ROOT::Reflex::Type;
    using ROOT::Reflex::Object;

     //------------------------------------------------------------------
    // Class template RefHolder<REF>
    //------------------------------------------------------------------


    template <class REF>
    class RefHolder : public RefHolderBase {
    public:
      RefHolder();
      explicit RefHolder(REF const& ref);
      RefHolder(RefHolder const& other);
      RefHolder& operator=(RefHolder const& rhs);
      void swap(RefHolder& other);
      virtual ~RefHolder();
      virtual RefHolderBase* clone() const;

      virtual ProductID id() const;
      virtual bool isEqualTo(RefHolderBase const& rhs) const;
      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;


      REF const& getRef() const;
      void setRef(REF const& r);
      virtual std::auto_ptr<RefVectorHolderBase> makeVectorHolder() const;

    private:
      virtual void const* pointerToType(Type const& iToType) const;
      REF ref_;
    };

    //------------------------------------------------------------------
    // Implementation of RefHolder<REF>
    //------------------------------------------------------------------

    template <class REF>
    RefHolder<REF>::RefHolder() : 
      ref_()
    { }
  
    template <class REF>
    RefHolder<REF>::RefHolder(RefHolder const& rhs) :
      ref_( rhs.ref_ )
    { }

    template <class REF>
    RefHolder<REF>& RefHolder<REF>::operator=(RefHolder const& rhs) {
      ref_ = rhs.ref_; return *this;
    }

    template <class REF>
    RefHolder<REF>::RefHolder(REF const& ref) : 
      ref_(ref) 
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
    RefHolder<REF>::pointerToType(Type const& iToType) const 
    {
      typedef typename REF::value_type contained_type;
      static const Type s_type(Type::ByTypeInfo(typeid(contained_type)));
    
      // The const_cast below is needed because
      // Object's constructor requires a pointer to
      // non-const void, although the implementation does not, of
      // course, modify the object to which the pointer points.
      Object obj(s_type, const_cast<void*>(static_cast<const void*>(ref_.get())));
      if ( s_type == iToType ) return obj.Address();
      Object cast = obj.CastObject(iToType);
      return cast.Address(); // returns void*, after pointer adjustment
    }
  } // namespace reftobase
}

#include "DataFormats/Common/interface/IndirectVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm {
  namespace reftobase {
    template <class REF>
    std::auto_ptr<RefVectorHolderBase> RefHolder<REF>::makeVectorHolder() const {
      typedef RefVector<typename REF::collection_type,
	                typename REF::value_type, 
                       	typename REF::finder_type> REFV;
      return std::auto_ptr<RefVectorHolderBase>( new RefVectorHolder<REFV> );
    }
  }
}

#endif
