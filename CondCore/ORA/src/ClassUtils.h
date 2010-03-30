#ifndef INCLUDE_ORA_CLASSUTILS_H
#define INCLUDE_ORA_CLASSUTILS_H

// externals
#include "Reflex/Type.h"

namespace ora {

  class RflxDeleter{

    public:
    RflxDeleter( const Reflex::Type& type );

    RflxDeleter( const RflxDeleter& rhs );

    ~RflxDeleter();

    void operator()( void* ptr );
    
  private:
    Reflex::Type m_type;
    
  };
    
  namespace ClassUtils {

    void* upCast( const Reflex::Type& type, void* ptr, const Reflex::Type& asType );

    bool isType( const Reflex::Type& type, const Reflex::Type& baseType );

    std::string demangledName( const std::type_info& typeInfo );

    Reflex::Type lookupDictionary( const std::type_info& typeInfo, bool throwFlag = true );

    Reflex::Type lookupDictionary( const std::string& className, bool throwFlag = true );

    void* constructObject( const Reflex::Type& typ );

    bool isTypeString(const Reflex::Type& typ);
    
    bool isTypePrimitive(const Reflex::Type& typ);
    
    bool isTypeContainer(const Reflex::Type& typ);

    bool isTypeKeyedContainer(const Reflex::Type& typ);

    bool isTypeNonKeyedContainer(const Reflex::Type& typ);

    bool isTypeAssociativeContainer(const Reflex::Type& typ);

    bool isTypeNonAssociativeContainer(const Reflex::Type& typ);

    Reflex::Type containerValueType(const Reflex::Type& typ);
    
    Reflex::Type containerKeyType(const Reflex::Type& typ);
    
    Reflex::Type containerDataType(const Reflex::Type& typ);
    
    Reflex::Type containerSubType(const Reflex::Type& typ, const std::string& subTypeName);
    
    Reflex::Type resolvedType(const Reflex::Type& typ);

    bool isTypeOraReference( const Reflex::Type& typ);

    bool isTypeOraPointer( const Reflex::Type& typ);
    
    bool isTypeUniqueReference( const Reflex::Type& typ);
    
    bool isTypePVector( const Reflex::Type& typ);

    bool isTypeQueryableVector( const Reflex::Type& typ);

    bool isTypeObject( const Reflex::Type& typ);

  }

}

#endif
