#ifndef Common_BaseHolder_h
#define Common_BaseHolder_h

namespace edm {
  class RefHolderBase;

  namespace reftobase {
    template<typename T> class BaseVectorHolder;

    //------------------------------------------------------------------
    // Class template BaseHolder<T>
    //
    // BaseHolder<T> is an abstract base class that manages a single
    // edm::Ref to an element of type T in a collection in the Event;
    // the purpose of this abstraction is to hide the type of the
    // collection from code that can not know about that type.
    // 
    //------------------------------------------------------------------
    template <class T>
    class BaseHolder {
    public:
      BaseHolder();
      virtual ~BaseHolder();
      virtual BaseHolder<T>* clone() const = 0;

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

    protected:
      // We want the following called only by derived classes.
      BaseHolder(BaseHolder const& other);
      BaseHolder& operator= (BaseHolder const& rhs);

    private:
    };

    //------------------------------------------------------------------
    // Implementation of BaseHolder<T>
    //------------------------------------------------------------------

    template <class T>
    BaseHolder<T>::BaseHolder() 
    { }

    template <class T>
    BaseHolder<T>::BaseHolder(BaseHolder const& other)
    {
      // Nothing to do.
    }

    template <class T>
    BaseHolder<T>&
    BaseHolder<T>::operator= (BaseHolder<T> const& other)
    {
      // No data to assign.
      return *this;
    }

    template <class T>
    BaseHolder<T>::~BaseHolder()
    {
      // nothing to do.
    }

  }
}

#endif
