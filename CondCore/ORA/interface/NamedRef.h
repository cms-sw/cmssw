#ifndef INCLUDE_ORA_NAMEDREF_H
#define INCLUDE_ORA_NAMEDREF_H

#include "Exception.h"
//
#include <assert.h>
#include <boost/shared_ptr.hpp>

namespace ora {
  
  class NamedReference {
    public:
    NamedReference();
    explicit NamedReference( const std::string& name );
    NamedReference( const std::string& name, boost::shared_ptr<void> ptr );
    NamedReference( const NamedReference& rhs );
    virtual ~NamedReference();
    NamedReference& operator=( const NamedReference& rhs );
    void set( const std::string& name );
    const std::string& name() const;
    bool isPersistent() const;
    boost::shared_ptr<void>& ptr() const;
    private:
    std::string m_name;
    bool m_isPersistent;
    mutable boost::shared_ptr<void> m_ptr;
  };

  template <typename T> class NamedRef : public NamedReference {
  public:
    NamedRef();
    NamedRef( const std::string& name );
    NamedRef( const std::string& name, boost::shared_ptr<T>& data );
    template <typename C> NamedRef( const std::string& name, boost::shared_ptr<C>& data );
    NamedRef( const NamedRef<T>& rhs );
    template <typename C> NamedRef( const NamedRef<C>& rhs );
    virtual ~NamedRef();
    NamedRef<T>& operator=( const NamedRef<T>& rhs );    
    template <typename C> NamedRef<T>& operator=( const NamedRef<C>& rhs );
    T* operator->() const;
    T& operator*() const;
    operator bool () const;
    T* get() const;
    boost::shared_ptr<T> share() const;
    bool operator!() const;
    bool operator==(const NamedRef<T>& rhs) const;
    bool operator!=(const NamedRef<T>& rhs) const;
    template <class C>
    bool operator==(const NamedRef<C>& rhs) const;
    template <class C>
    bool operator!=(const NamedRef<C>& rhs) const;
  private:
    T* safePtr( bool throw_flag ) const;

  };
  
}

template <class T>
inline ora::NamedRef<T>::NamedRef() :
  NamedReference(){}

template <class T>
inline ora::NamedRef<T>::NamedRef( const std::string& name ) :
  NamedReference( name ){}

template <class T>
inline ora::NamedRef<T>::NamedRef( const std::string& name, boost::shared_ptr<T>& data ) :
  NamedReference( name, boost::shared_ptr<void>( data ) ){}

template <class T>
template <class C>
inline ora::NamedRef<T>::NamedRef( const std::string& name, boost::shared_ptr<C>& data ) :
  NamedReference( name, boost::shared_ptr<void>( data )){
}

template <class T>
inline ora::NamedRef<T>::NamedRef(const NamedRef<T>& rhs) :
  NamedReference( rhs.name() ){
}

template <class T>
template <class C>
inline ora::NamedRef<T>::NamedRef(const NamedRef<C>& rhs) :
  NamedReference( rhs.name() ){
}

template <class T>
inline ora::NamedRef<T>::~NamedRef() {
}

template <class T>
inline ora::NamedRef<T>& ora::NamedRef<T>::operator=(const NamedRef<T>& rhs ){
  NamedReference::operator=( rhs );
  return *this;  
}

template <class T> 
template <class C>
inline ora::NamedRef<T>& ora::NamedRef<T>::operator=(const NamedRef<C>&  rhs ){
  NamedReference::operator=( rhs );
  return *this;  
}

template <class T>
T* ora::NamedRef<T>::safePtr( bool throw_flag ) const {
  T* p = share().get();
  if( !p && throw_flag) throwException( "Underlying pointer is null.","NamedRef::safePtr"); 
  return p;
}

template <class T>
inline T* ora::NamedRef<T>::operator->() const {
  return safePtr( true );
}

template <class T>
inline T& ora::NamedRef<T>::operator*() const {
  return *safePtr( true );
}

template <class T>
inline ora::NamedRef<T>::operator bool() const {
  return safePtr(false);
}

template <class T>
inline T* ora::NamedRef<T>::get() const  {
  return safePtr(false);  
}

template <class T>
inline boost::shared_ptr<T> ora::NamedRef<T>::share() const {
  return boost::static_pointer_cast<T>(ptr());
}

template <class T>
inline bool ora::NamedRef<T>::operator!() const {
  return safePtr(false)==0;
}

template <class T>
bool ora::NamedRef<T>::operator==(const NamedRef<T>& rhs) const {
  return share() == rhs.share();
}

template <class T>
bool ora::NamedRef<T>::operator!=(const NamedRef<T>& rhs ) const {
  return !(this->operator==(rhs));     
}

template <class T>
template <class C>
bool ora::NamedRef<T>::operator==(const NamedRef<C>& rhs) const {
  return share() == rhs.share();
}

template <class T>
template <class C>
bool ora::NamedRef<T>::operator!=(const NamedRef<C>& rhs ) const {
  return !(this->operator==(rhs));     
}

#endif
  
    
    
