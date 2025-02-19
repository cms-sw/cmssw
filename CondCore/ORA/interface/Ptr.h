#ifndef INCLUDE_ORA_PTR_H
#define INCLUDE_ORA_PTR_H

#include "Exception.h"
//
#include <assert.h>
#include <typeinfo>
#include <boost/shared_ptr.hpp>

namespace ora {

   /**
   * @class IPtrLoader Ptr.h
   *  
   * Interface for the lazy loading of the embedded object.
   * Implementations are provided by the db storage code.
   */
  class IPtrLoader {
    
    public:
    // destructor
    virtual ~IPtrLoader(){}

    public:
    // triggers the data loading
    virtual void* load() const=0;

    // notify the underlying storage system that the embedded object has been destructed.
    // maybe not required.
    //virtual void notify() const=0;

    // invalidates the current loader. Called by the underlying service at his destruction time.
    virtual void invalidate()=0;

    // queries the validity of the current relation with the underlying storage system
    virtual bool isValid() const=0;

  };
  
   /**
   * @class Ptr Ptr.h 
   *  
   * Templated class for the persistency of associated objects. The embedded object is treated as  
   * a normal C++ pointer in the writing mode, without to require an explicit write request on it.
   * In the reading mode, the embedded object is loaded only at access time (lazy loading).
   */
  template <typename T> class Ptr {

    public:
    
    // default constructor
    Ptr();
    
    // from a real pointer
    explicit Ptr(T* anObject);
    
    // copy constructor
    Ptr(const Ptr<T>&); 

    // extended copy constructor
    template <class C> Ptr(const Ptr<C>&);  

    // destructor
    virtual ~Ptr();

    // Assignment operator with real pointer 
    Ptr<T>& operator=(T*);

    // assignment operator
    Ptr<T>& operator=(const Ptr<T>&);

    // extended assignment operator 
    template <class C> Ptr<T>& operator=(const Ptr<C>&);

    // copy operator for up/down casting 
    template <class C> Ptr<T>& cast(const Ptr<C>&);

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

    // 'not' operator for consistency with pointers common use
    bool operator!() const;

    // equality operator 
    template <class C>
    bool operator==(const Ptr<C>& aPtr) const {
      return m_ptr == static_cast<C*>(aPtr.address());
    }
    template <class C>
    bool operator!=(const Ptr<C>& aPtr) const {
      return !(this->operator==(aPtr));     
    }
    
    public:

    // clear the embedded pointer and invalidates the loader,if any.
    void reset();

    // returns the current loader
    boost::shared_ptr<IPtrLoader>& loader() const {
      return m_loader;
    }

    // triggers the loading if the loader is installed
    void load() const;

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

template <class T>
inline ora::Ptr<T>::Ptr() :
  m_ptr(),m_loader(),m_isLoaded(false) {}

template <class T>
inline ora::Ptr<T>::Ptr(T* anObject) :
  m_ptr(anObject),m_loader(),m_isLoaded(true) {}

template <class T>
inline ora::Ptr<T>::Ptr(const Ptr<T>& aPtr) :
  m_ptr(aPtr.m_ptr),m_loader(aPtr.m_loader),m_isLoaded(false){
}

template <class T>
template <class C>
inline ora::Ptr<T>::Ptr(const Ptr<C>& aPtr) :
  m_ptr(aPtr.share()),m_loader(aPtr.loader()),m_isLoaded(aPtr.isLoaded()) {
  // compile-time type checking
  C* c = 0; T* t(c); assert(t==0);
}

template <class T>
inline ora::Ptr<T>::~Ptr() {
}

template <class T>
inline ora::Ptr<T>& ora::Ptr<T>::operator=(T* aPtr) {
  m_ptr.reset(aPtr);
  m_isLoaded = true;
  return *this;
}

template <class T>
inline ora::Ptr<T>& ora::Ptr<T>::operator=(const Ptr<T>& aPtr){
  m_loader = aPtr.m_loader;
  m_isLoaded = aPtr.m_isLoaded;
  m_ptr = aPtr.m_ptr;
  return *this;  
}

template <class T> 
template <class C>
inline ora::Ptr<T>& ora::Ptr<T>::operator=(const Ptr<C>& aPtr){
  C* c = 0; T* t(c); assert(t==0);
  m_loader = aPtr.loader();
  m_isLoaded = aPtr.isLoaded();
  m_ptr = aPtr.share();
  return *this;
}

template <class T>
template <class C>
inline ora::Ptr<T>& ora::Ptr<T>::cast(const Ptr<C>& aPtr){
  m_loader = aPtr.loader();
  m_ptr = boost::dynamic_pointer_cast( aPtr.share());
  m_isLoaded = aPtr.isLoaded();
  return *this;  
}

template <class T>
inline T* ora::Ptr<T>::operator->() const {
  return ptr(true);
}

template <class T>
inline T& ora::Ptr<T>::operator*() const {
  return *ptr(true);
}

template <class T>
inline T* ora::Ptr<T>::get() const  {
  return ptr(false);  
}

template <class T>
inline boost::shared_ptr<T>& ora::Ptr<T>::share() const {
  return m_ptr;
}

template <class T>
inline void* ora::Ptr<T>::address() const  {
  return m_ptr.get();  
}

template <class T>
inline ora::Ptr<T>::operator bool() const {
  return ptr(false);
}

template <class T>
inline bool ora::Ptr<T>::operator!() const {
  return ptr(false)==0;
}

template <class T>
inline void ora::Ptr<T>::reset(){
  m_ptr.reset();
  m_loader.reset();
  m_isLoaded = false;
}

template <class T>
inline void ora::Ptr<T>::load() const {
  ptr( false );
}

template <class T>
inline T* ora::Ptr<T>::ptr(bool throwFlag) const {
  if(!m_ptr.get()){
    if(!m_loader.get()){
      if(throwFlag) throwException("Loader is not installed.","Ptr::ptr()");
    }
    if(!m_isLoaded && m_loader.get()){
      m_ptr.reset( static_cast<T*>(m_loader->load()));
      m_isLoaded = true;
    }
  }
  if(!m_ptr.get()){
    if(throwFlag) throwException("Underlying pointer is null.","Ptr::ptr()");    
  }
  return m_ptr.get();
}

#endif 
