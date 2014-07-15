#ifndef INCLUDE_ORA_OBJECT_H
#define INCLUDE_ORA_OBJECT_H

//
#include <typeinfo>
#include <boost/shared_ptr.hpp>
// externals
#include "Reflex/Type.h"

namespace ora {
  
  class Object {
    public:
    Object();
    template <typename T>
    explicit Object( const T& obj );
    Object( const void* ptr, const std::type_info& typeInfo );
    Object( const void* ptr, const Reflex::Type& type );
    Object( const void* ptr, const std::string& typeName );
    Object( const Object& rhs);
    virtual ~Object();
    Object& operator=( const Object& rhs);
    bool operator==( const Object& rhs) const;
    bool operator!=( const Object& rhs) const;
    void* address() const;
    const Reflex::Type& type() const;
    std::string typeName() const;
    void* cast( const std::type_info& asType ) const;
    template <typename T> T* cast() const;
    boost::shared_ptr<void> makeShared() const;
    void destruct();
    private:
    void* m_ptr;
    Reflex::Type m_type;
  };

  template<> 
  inline
  Object::Object( const Object& rhs ):
    m_ptr( rhs.m_ptr ),
    m_type( rhs.m_type ){
  }

}

template <typename T>
inline
ora::Object::Object( const T& obj ):
  Object( &obj, typeid(obj) ){
}

template <typename T>
inline
T* 
ora::Object::cast() const {
  const std::type_info& typeInfo = typeid(T);
  return static_cast<T*>( cast( typeInfo ) );
}

#endif

