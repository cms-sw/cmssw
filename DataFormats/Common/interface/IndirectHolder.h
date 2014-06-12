#ifndef DataFormats_Common_IndirectHolder_h
#define DataFormats_Common_IndirectHolder_h
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/BaseHolder.h"
#include "DataFormats/Common/interface/RefHolderBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

#include <memory>

namespace edm {
  template<typename T> class RefToBase;

  namespace reftobase {

    template<typename T> class IndirectVectorHolder;
    class RefVectorHolderBase;
    class RefHolderBase;

    //------------------------------------------------------------------
    // Class template IndirectHolder<T>
    //------------------------------------------------------------------

    template <typename T>
    class IndirectHolder : public BaseHolder<T> {
    public:
      // It may be better to use auto_ptr<RefHolderBase> in
      // this constructor, so that the cloning can be avoided. I'm not
      // sure if use of auto_ptr here causes any troubles elsewhere.
      IndirectHolder() : BaseHolder<T>(), helper_( 0 ) { }
      IndirectHolder(std::shared_ptr<RefHolderBase> p);
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
      virtual std::auto_ptr<RefVectorHolderBase> makeVectorBaseHolder() const;
      virtual EDProductGetter const* productGetter() const;
      virtual bool hasProductCache() const;
      virtual void const * product() const;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const { return helper_->isAvailable(); }

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
    inline
    IndirectHolder<T>::IndirectHolder(std::shared_ptr<RefHolderBase> p) :
      BaseHolder<T>(), helper_(p->clone()) 
    { }

    template <typename T>
    inline
    IndirectHolder<T>::IndirectHolder(IndirectHolder const& other) : 
      BaseHolder<T>(other), helper_(other.helper_->clone()) 
    { }

    template <typename T>
    inline
    void
    IndirectHolder<T>::swap(IndirectHolder& other) 
    {
      this->BaseHolder<T>::swap(other);
      std::swap(helper_, other.helper_);
    }

    template <typename T>
    inline
    IndirectHolder<T>& 
    IndirectHolder<T>::operator= (IndirectHolder const& rhs) 
    {
      IndirectHolder temp(rhs);
      swap(temp);
      return *this;
    }

    template <typename T>
    IndirectHolder<T>::~IndirectHolder()
    {
      delete helper_;
    }

    template <typename T>
    BaseHolder<T>* 
    IndirectHolder<T>::clone() const
    {
      return new IndirectHolder<T>(*this);
    }

    template <typename T>
    T const* 
    IndirectHolder<T>::getPtr() const 
    {
     return helper_-> template getPtr<T>();
    }

    template <typename T>
    ProductID
    IndirectHolder<T>::id() const
    {
      return helper_->id();
    }

    template <typename T>
    size_t
    IndirectHolder<T>::key() const
    {
      return helper_->key();
    }

    template <typename T>
    inline
    EDProductGetter const* IndirectHolder<T>::productGetter() const {
      return helper_->productGetter();
    }

    template <typename T>
    inline
    bool IndirectHolder<T>::hasProductCache() const {
      return helper_->hasProductCache();
    }

    template <typename T>
    inline
    void const * IndirectHolder<T>::product() const {
      return helper_->product();
    }

    template <typename T>
    bool
    IndirectHolder<T>::isEqualTo(BaseHolder<T> const& rhs) const 
    {
      IndirectHolder const* h = dynamic_cast<IndirectHolder const*>(&rhs);
      return h && helper_->isEqualTo(*h->helper_);
    }

    template <typename T>
    bool
    IndirectHolder<T>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					      std::string& msg) const
    {
      return helper_->fillRefIfMyTypeMatches(fillme, msg);
    }

    template <typename T>
    std::auto_ptr<RefHolderBase> IndirectHolder<T>::holder() const { 
      return std::auto_ptr<RefHolderBase>( helper_->clone() ); 
    }

    // Free swap function
    template <typename T>
    inline
    void
    swap(IndirectHolder<T>& lhs, IndirectHolder<T>& rhs) {
      lhs.swap(rhs);
    }
  }

}

#include "DataFormats/Common/interface/IndirectVectorHolder.h"
#include "DataFormats/Common/interface/RefVectorHolderBase.h"

namespace edm {
  namespace reftobase {
    template <typename T>
    std::auto_ptr<BaseVectorHolder<T> > IndirectHolder<T>::makeVectorHolder() const {
      std::auto_ptr<RefVectorHolderBase> p = helper_->makeVectorHolder();
      std::shared_ptr<RefVectorHolderBase> sp( p.release() );
      return std::auto_ptr<BaseVectorHolder<T> >( new IndirectVectorHolder<T>( sp ) );
    }

    template <typename T>
    std::auto_ptr<RefVectorHolderBase> IndirectHolder<T>::makeVectorBaseHolder() const {
      return helper_->makeVectorHolder();
    }
  }
}

#endif
