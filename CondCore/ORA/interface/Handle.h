#ifndef INCLUDE_ORA_HANDLE_H
#define INCLUDE_ORA_HANDLE_H

#include "CondCore/ORA/interface/Exception.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>


namespace ora {

  template <typename T> class Holder {
    public:
    Holder();
    Holder( T* p );
    ~Holder();
    std::auto_ptr<T> ptr;
  };
  
  template <typename T> class Handle {
    public:
    Handle();

    explicit Handle(T* ptr);

    Handle(const Handle<T>& rhs);

    ~Handle();

    Handle& operator=(const Handle<T>& rhs);

    void reset( T* ptr );

    // dereference operator
    T* operator->() const;

    // dereference operator
    T& operator*() const;

    // implicit bool conversion
    operator bool () const;

    // not operator
    bool operator!() const;

    // return the real pointer
    T* get() const;

    void clear();
    private:
    void validate() const;

    private:
    boost::shared_ptr< Holder<T> > m_holder;
  };
}

template <typename T>
inline
ora::Holder<T>::Holder():
  ptr(){
}

template <typename T>
inline
ora::Holder<T>::Holder(T* p ):
  ptr(p){
}

template <typename T>
inline
ora::Holder<T>::~Holder(){
}

template <typename T>
inline
ora::Handle<T>::Handle():
  m_holder( new Holder<T>() ){
}

template <typename T>
inline
ora::Handle<T>::Handle(T* ptr):
  m_holder( new Holder<T>( ptr ) ){
}

template <typename T>
inline
ora::Handle<T>::Handle( const Handle<T>& rhs ):
  m_holder( rhs.m_holder ){
}

template <typename T>
inline
ora::Handle<T>::~Handle(){
}

template <typename T>
inline
ora::Handle<T>& ora::Handle<T>::operator=( const Handle<T>& rhs ){
  m_holder = rhs.m_holder;
  return *this;
}

template <typename T>
inline
void ora::Handle<T>::reset( T* ptr ){
  m_holder.reset( new Holder<T>( ptr ) );
}

template <typename T>
inline
void ora::Handle<T>::validate() const{
  if(!m_holder->ptr.get()) {
    throwException( "Handle is not valid.",typeid(Handle<T>),"validate");
  }
}

template <typename T>
inline
T* ora::Handle<T>::operator->() const {
  validate();
  return m_holder->ptr.get();
}

template <typename T>
inline
T& ora::Handle<T>::operator*() const {
  validate();
  return *m_holder->ptr.get();
}

template <typename T>
inline
ora::Handle<T>::operator bool () const {
  return m_holder->ptr.get()!=0;  
}

template <typename T>
inline
bool ora::Handle<T>::operator!() const {
  return m_holder->ptr.get()==0;  
}

template <typename T>
inline
T* ora::Handle<T>::get() const {
  return m_holder->ptr.get();  
}

template <typename T>
inline
void ora::Handle<T>::clear(){
  if(m_holder.get()) m_holder->ptr.reset();
} 

#endif 


