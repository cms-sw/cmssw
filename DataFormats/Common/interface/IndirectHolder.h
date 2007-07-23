#ifndef Common_IndirectHolder_h
#define Common_IndirectHolder_h
#include "DataFormats/Common/interface/BaseHolder.h"
#include "DataFormats/Common/interface/RefHolderBase.h"

namespace edm {
  template<typename T> class RefToBase;

  namespace reftobase {

    template<typename T> class IndirectVectorHolder;

    class RefHolderBase;

    //------------------------------------------------------------------
    // Class template IndirectHolder<T>
    //------------------------------------------------------------------

    template <class T>
    class IndirectHolder : public BaseHolder<T> {
    public:
      // It may be better to use auto_ptr<RefHolderBase> in
      // this constructor, so that the cloning can be avoided. I'm not
      // sure if use of auto_ptr here causes any troubles elsewhere.
      IndirectHolder() : helper_( 0 ) { }
      IndirectHolder(boost::shared_ptr<RefHolderBase> p);
      IndirectHolder(IndirectHolder const& other);
      IndirectHolder& operator= (IndirectHolder const& rhs);
      void swap(IndirectHolder& other);
      virtual ~IndirectHolder();
      
      virtual BaseHolder<T>* clone() const;
      virtual T const* getPtr() const;
      virtual ProductID id() const;
      virtual size_t key() const;
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const;

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;
      virtual std::auto_ptr<RefHolderBase> holder() const;
      virtual std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() const;

    private:
      friend class RefToBase<T>;
      friend class IndirectVectorHolder<T>;
      RefHolderBase* helper_;
    };
    //------------------------------------------------------------------
    // Implementation of IndirectHolder<T>
    //------------------------------------------------------------------

    template <class T>
    inline
    IndirectHolder<T>::IndirectHolder(boost::shared_ptr<RefHolderBase> p) :
      helper_(p->clone()) 
    { }

    template <class T>
    inline
    IndirectHolder<T>::IndirectHolder(IndirectHolder const& other) : 
      helper_(other.helper_->clone()) 
    { }

    template <class T>
    inline
    IndirectHolder<T>& 
    IndirectHolder<T>::operator= (IndirectHolder const& rhs) 
    {
      IndirectHolder temp(rhs);
      swap(temp);
      return *this;
    }

    template <class T>
    inline
    void
    IndirectHolder<T>::swap(IndirectHolder& other) 
    {
      std::swap(helper_, other.helper_);
    }

    template <class T>
    IndirectHolder<T>::~IndirectHolder()
    {
      delete helper_;
    }

    template <class T>
    BaseHolder<T>* 
    IndirectHolder<T>::clone() const
    {
      return new IndirectHolder<T>(*this);
    }

    template <class T>
    T const* 
    IndirectHolder<T>::getPtr() const 
    {
     return helper_-> template getPtr<T>();
    }

    template <class T>
    ProductID
    IndirectHolder<T>::id() const
    {
      return helper_->id();
    }

    template <class T>
    size_t
    IndirectHolder<T>::key() const
    {
      return helper_->key();
    }

    template <class T>
    bool
    IndirectHolder<T>::isEqualTo(BaseHolder<T> const& rhs) const 
    {
      IndirectHolder const* h = dynamic_cast<IndirectHolder const*>(&rhs);
      return h && helper_->isEqualTo(*h->helper_);
    }

    template <class T>
    bool
    IndirectHolder<T>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					      std::string& msg) const
    {
      return helper_->fillRefIfMyTypeMatches(fillme, msg);
    }

    template <class T>
    std::auto_ptr<RefHolderBase> IndirectHolder<T>::holder() const { 
      return std::auto_ptr<RefHolderBase>( helper_->clone() ); 
    }
  }
}

#include "DataFormats/Common/interface/IndirectVectorHolder.h"

namespace edm {
  namespace reftobase {
    template <class T>
    std::auto_ptr<BaseVectorHolder<T> > IndirectHolder<T>::makeVectorHolder() const {
      std::auto_ptr<RefVectorHolderBase> p = helper_->makeVectorHolder();
      boost::shared_ptr<RefVectorHolderBase> sp( p );
      return std::auto_ptr<BaseVectorHolder<T> >( new IndirectVectorHolder<T>( sp ) );
    }
  }
}

#endif
