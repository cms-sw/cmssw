#ifndef INCLUDE_ORA_CLASSUTILS_H
#define INCLUDE_ORA_CLASSUTILS_H

// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace ora {

  class RflxDeleter{

    public:
    RflxDeleter( const edm::TypeWithDict& type );

    RflxDeleter( const RflxDeleter& rhs );

    ~RflxDeleter();

    void operator()( void* ptr );
    
  private:
    edm::TypeWithDict m_type;
    
  };
    
  namespace ClassUtils {

    void loadDictionary(  const std::string& className );

    void* upCast( const edm::TypeWithDict& type, void* ptr, const edm::TypeWithDict& asType );

    bool isType( const edm::TypeWithDict& type, const edm::TypeWithDict& baseType );

    bool checkMappedType( const edm::TypeWithDict& type, const std::string& mappedTypeName );

    bool findBaseType( edm::TypeWithDict& type, edm::TypeWithDict& baseType, size_t& func );

    std::string demangledName( const std::type_info& typeInfo );

    edm::TypeWithDict lookupDictionary( const std::type_info& typeInfo, bool throwFlag = true );

    edm::TypeWithDict lookupDictionary( const std::string& className, bool throwFlag = true );

    void* constructObject( const edm::TypeWithDict& typ );

    bool isTypeString(const edm::TypeWithDict& typ);
    
    bool isTypePrimitive(const edm::TypeWithDict& typ);
    
    bool isTypeContainer(const edm::TypeWithDict& typ);

    bool isTypeKeyedContainer(const edm::TypeWithDict& typ);

    bool isTypeNonKeyedContainer(const edm::TypeWithDict& typ);

    bool isTypeAssociativeContainer(const edm::TypeWithDict& typ);

    bool isTypeNonAssociativeContainer(const edm::TypeWithDict& typ);

    edm::TypeWithDict containerValueType(const edm::TypeWithDict& typ);
    
    edm::TypeWithDict containerKeyType(const edm::TypeWithDict& typ);
    
    edm::TypeWithDict containerDataType(const edm::TypeWithDict& typ);
    
    edm::TypeWithDict containerSubType(const edm::TypeWithDict& typ, const std::string& subTypeName);
    
    edm::TypeWithDict resolvedType(const edm::TypeWithDict& typ);

    bool isTypeOraReference( const edm::TypeWithDict& typ);

    bool isTypeNamedReference( const edm::TypeWithDict& typ);

    bool isTypeOraPointer( const edm::TypeWithDict& typ);
    
    bool isTypeUniqueReference( const edm::TypeWithDict& typ);
    
    bool isTypePVector( const edm::TypeWithDict& typ);

    bool isTypeQueryableVector( const edm::TypeWithDict& typ);

    bool isTypeOraVector( const edm::TypeWithDict& typ);

    bool isTypeNameOraVector( const std::string& typeName );

    bool isTypeObject( const edm::TypeWithDict& typ);

    size_t arrayLength( const edm::TypeWithDict& typ );

    std::string getClassProperty( const std::string& propertyName, const edm::TypeWithDict& type );
    std::string getDataMemberProperty( const std::string& propertyName, const edm::MemberWithDict& dataMember );

  }

}

#endif
