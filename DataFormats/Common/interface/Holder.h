#ifndef Common_Holder_h
#define Common_Holder_h
#include "DataFormats/Common/interface/BaseHolder.h"
#include "DataFormats/Common/interface/RefHolder.h"

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
      virtual BaseHolder<T>* clone() const;

      virtual T const* getPtr() const;
      virtual ProductID id() const;
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const;
      REF const& getRef() const;

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;

      virtual std::auto_ptr<RefHolderBase> holder() const {
	return std::auto_ptr<RefHolderBase>( new RefHolder<REF>( ref_ ) );
      }
      virtual std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() const;

    private:
      REF ref_;
    };

    //------------------------------------------------------------------
    // Implementation of Holder<T,REF>
    //------------------------------------------------------------------

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder() : 
      ref_()
    {  }

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder(Holder const& other) :
      ref_(other.ref_)
    { }

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder(REF const& r) :
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

#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace edm {
  namespace reftobase {
    template <class T, class REF>
    std::auto_ptr<BaseVectorHolder<T> > Holder<T,REF>::makeVectorHolder() const {
      typedef RefVector<typename REF::collection_type,
	                typename REF::value_type, 
                       	typename REF::finder_type> REFV;
      return std::auto_ptr<BaseVectorHolder<T> >( new VectorHolder<T, REFV> );
    }
  }
}

#endif
