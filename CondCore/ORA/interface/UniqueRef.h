#ifndef INCLUDE_ORA_UNIQUEREF_H
#define INCLUDE_ORA_UNIQUEREF_H

#include "Ptr.h"

namespace ora {

  /**
   * @class UniqueRef UniqueRef.h
   *  
   * Templated class for the persistency of associated objects. Same features as Ptr (lazy loading) + support of polymorhism .
   */
  template <typename T> class UniqueRef {

    public:
    
    // default constructor
    UniqueRef();
    
    // from a real pointer
    explicit UniqueRef(T* anObject);
    
    // copy constructor
    UniqueRef(const UniqueRef<T>&); 

    // extended copy constructor
    template <class C> UniqueRef(const UniqueRef<C>&);  

    // destructor
    virtual ~UniqueRef();

    // Assignment operator with real pointer 
    UniqueRef<T>& operator=(T*);

    // assignment operator
    UniqueRef<T>& operator=(const UniqueRef<T>&);

    // extended assignment operator 
    template <class C> UniqueRef<T>& operator=(const UniqueRef<C>&);

    // copy operator for up/down casting 
    template <class C> UniqueRef<T>& cast(const UniqueRef<C>&);

    // dereference operator
    T* operator->() const;

    // dereference operator
    T& operator*() const;

    // implicit bool conversion
    operator bool () const;

    // return the real pointer
    T* get() const;

    // return the shared ptr
    boost::shared_ptr<T>& share() const;

    // return the naked pointer, without to trigger the loading.
    void* address() const;

    // return the real ptr type
    const std::type_info* typeInfo() const;
      
    // 'not' operator for consistency with pointers common use
    bool operator!() const;

    // equality operator
    template <class C>
    bool operator==(const UniqueRef<C>& aPtr) const {
      return m_ptr == static_cast<C*>(aPtr.address());
    }
    template <class C>
    bool operator!=(const UniqueRef<C>& aPtr) const {
      return !(this->operator==(aPtr));     
    }
    
    public:

    // clear the embedded pointer and invalidates the loader,if any.
    void reset();

    // returns the current loader
    boost::shared_ptr<IPtrLoader>& loader() const {
      return m_loader;
    }

    // returns true if the emebedded object data have been loaded.
    bool isLoaded() const {
      return m_isLoaded;
    }
    
    private:
    
    // pointer with throw exception clause
    T* ptr(bool throw_flag) const;

  private:

    // embedded object pointer
    mutable boost::shared_ptr<T> m_ptr;

    // data loader, istalled by the storage system
    mutable boost::shared_ptr<IPtrLoader> m_loader;

    // object loaded flag
    mutable bool m_isLoaded;

  };
  
  
}

template <class T> ora::UniqueRef<T>::UniqueRef() :
  m_ptr(),m_loader(),m_isLoaded(false) {}

template <class T> ora::UniqueRef<T>::UniqueRef(T* anObject) :
  m_ptr(anObject),m_loader(),m_isLoaded(true) {}

template <class T> ora::UniqueRef<T>::UniqueRef(const UniqueRef<T>& aPtr) :
  m_ptr(aPtr.m_ptr),m_loader(aPtr.m_loader),m_isLoaded(false){
}

template <class T> 
template <class C> ora::UniqueRef<T>::UniqueRef(const UniqueRef<C>& aPtr) :
  m_ptr(aPtr.share()),m_loader(aPtr.loader()),m_isLoaded(aPtr.isLoaded()) {
  // compile-time type checking
  C* c = 0; T* t(c); assert(t==0);
}

template <class T> ora::UniqueRef<T>::~UniqueRef() {
}

template <class T> ora::UniqueRef<T>& ora::UniqueRef<T>::operator=(T* aPtr) {
  reset();
  m_ptr.reset(aPtr);
  m_isLoaded = true;
  return *this;
}

template <class T> ora::UniqueRef<T>& ora::UniqueRef<T>::operator=(const UniqueRef<T>& aPtr){
  reset();
  m_loader = aPtr.m_loader;
  m_ptr = aPtr.m_ptr;
  m_isLoaded = aPtr.m_isLoaded;
  return *this;  
}

template <class T> 
template <class C> ora::UniqueRef<T>& ora::UniqueRef<T>::operator=(const UniqueRef<C>& aPtr){
  C* c = 0; T* t(c); assert(t==0);
  reset();
  m_loader = aPtr.loader();
  m_ptr = aPtr.share();
  m_isLoaded = aPtr.isLoaded();
  return *this;  
}

template <class T> template <class C> ora::UniqueRef<T>& ora::UniqueRef<T>::cast(const UniqueRef<C>& aPtr){
  reset();
  m_loader = aPtr.loader();
  m_ptr = boost::dynamic_pointer_cast(aPtr.share());
  m_isLoaded = aPtr.isLoaded();
  return *this;  
}

template <class T> T* ora::UniqueRef<T>::operator->() const {
  return ptr(true);
}

template <class T> T& ora::UniqueRef<T>::operator*() const {
  return *ptr(true);
}

template <class T> T* ora::UniqueRef<T>::get() const  {
  return ptr(false);  
}

template <class T>
inline boost::shared_ptr<T>& ora::UniqueRef<T>::share() const {
  return m_ptr;
}

template <class T> void* ora::UniqueRef<T>::address() const  {
  return m_ptr.get();  
}

template <class T> const std::type_info* ora::UniqueRef<T>::typeInfo() const  {
  const std::type_info* ret = 0;
  if(m_ptr) ret = &typeid(*m_ptr);
  return ret;
}

template <class T> ora::UniqueRef<T>::operator bool() const {
  return ptr(false);
}

template <class T> bool ora::UniqueRef<T>::operator!() const {
  return ptr(false)==0;
}

template <class T> void ora::UniqueRef<T>::reset(){
  m_ptr.reset();
  m_loader.reset();
  m_isLoaded = false;
}

template <class T> T* ora::UniqueRef<T>::ptr(bool throwFlag) const {
  if(!m_ptr.get()){
    if(!m_loader.get()){
      if(throwFlag) throwException("Loader is not installed.",
                                   "UniqueRef::ptr()");
    }
    if(!m_isLoaded && m_loader.get()){
      m_ptr.reset( static_cast<T*>(m_loader->load()));
      m_isLoaded = true;
    }
  }
  if(!m_ptr.get()){
    if(throwFlag) throwException("Underlying pointer is null.",
                                 "UniqueRef::ptr()");    
  }
  return m_ptr.get();
}

#endif
