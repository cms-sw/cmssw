#ifndef DataFormats_Common_BaseHolder_h
#define DataFormats_Common_BaseHolder_h

#include "DataFormats/Common/interface/EDProductGetter.h"
#include <string>
#include <memory>

namespace edm {
  class ProductID;
  class RefHolderBase;

  namespace reftobase {
    template<typename T> class BaseVectorHolder;
    class RefVectorHolderBase;

    //------------------------------------------------------------------
    // Class template BaseHolder<T>
    //
    // BaseHolder<T> is an abstract base class that manages a single
    // edm::Ref to an element of type T in a collection in the Event;
    // the purpose of this abstraction is to hide the type of the
    // collection from code that can not know about that type.
    // 
    //------------------------------------------------------------------
    template <typename T>
    class BaseHolder {
    public:
      BaseHolder();
      virtual ~BaseHolder();
      virtual BaseHolder<T>* clone() const = 0;

      void swap(BaseHolder&);

      // Return the address of the element to which the hidden Ref
      // refers.
      virtual T const* getPtr() const = 0;

      // Return the ProductID of the collection to which the hidden
      // Ref refers.
      virtual ProductID id() const = 0;
      virtual size_t key() const = 0;
      // Check to see if the Ref hidden in 'rhs' is equal to the Ref
      // hidden in 'this'. They can not be equal if they are of
      // different types. Note that the equality test also returns
      // false if dynamic type of 'rhs' is different from the dynamic
      // type of 'this', *even when the hiddens Refs are actually
      // equivalent*.
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const = 0;

      // If the type of Ref I contain matches the type contained in
      // 'fillme', set the Ref in 'fillme' equal to mine and return
      // true. If not, write the name of the type I really contain to
      // msg, and return false.
      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const = 0;
      virtual std::auto_ptr<RefHolderBase> holder() const = 0;

      virtual std::auto_ptr<BaseVectorHolder<T> > makeVectorHolder() const = 0;
      virtual std::auto_ptr<RefVectorHolderBase> makeVectorBaseHolder() const = 0;

      virtual EDProductGetter const* productGetter() const = 0;
      virtual bool hasProductCache() const = 0;
      virtual void const * product() const = 0;

      /// Checks if product collection is in memory or available
      /// in the Event. No type checking is done.
      virtual bool isAvailable() const = 0;

    protected:
      // We want the following called only by derived classes.
      BaseHolder(BaseHolder const& other);
      BaseHolder& operator= (BaseHolder const& rhs);

    private:
    };

    //------------------------------------------------------------------
    // Implementation of BaseHolder<T>
    //------------------------------------------------------------------

    template <typename T>
    BaseHolder<T>::BaseHolder() 
    { }

    template <typename T>
    BaseHolder<T>::BaseHolder(BaseHolder const& other)
    {
      // Nothing to do.
    }

    template <typename T>
    BaseHolder<T>&
    BaseHolder<T>::operator= (BaseHolder<T> const& other)
    {
      // No data to assign.
      return *this;
    }

    template <typename T>
    BaseHolder<T>::~BaseHolder()
    {
      // nothing to do.
    }

    template <typename T>
    void
    BaseHolder<T>::swap(BaseHolder<T>& other) {
      // nothing to do.
    }

    // Free swap function
    template <typename T>
    inline
    void
    swap(BaseHolder<T>& lhs, BaseHolder<T>& rhs) {
      lhs.swap(rhs);
    }
  }
}

#endif
