#ifndef INCLUDE_ORA_RELATIONALMAPPING_H
#define INCLUDE_ORA_RELATIONALMAPPING_H

//
#include <utility>
#include <string>
// externals
#include "Reflex/Type.h"

namespace Reflex {
  class Type;
}

namespace ora {

  namespace RelationalMapping {
    size_t sizeInColumns(const Reflex::Type& topLevelClassType );
    std::pair<bool,size_t> sizeInColumnsForCArray( const Reflex::Type& arrayType );
    void _sizeInColumns(const Reflex::Type& typ, size_t& sz, bool& hasDependencies );
    void _sizeInColumnsForCArray(const Reflex::Type& typ, size_t& sz, bool& hasDependencies );
  }

  class MappingElement;
  class TableRegister;
  
  class IRelationalMapping {
    public:
    virtual ~IRelationalMapping(){
    }

    virtual void process( MappingElement& parentElement, const std::string& attributeName,
                          const std::string& attributeNameForSchema, const std::string& scopeNameForSchema ) = 0;
  };

  class RelationalMappingFactory {
    public:
    explicit RelationalMappingFactory( TableRegister& tableRegister );
    virtual ~RelationalMappingFactory();

    public:
    IRelationalMapping* newProcessor( const Reflex::Type& attributeType, bool blobStreaming=false );

    private:
    TableRegister& m_tableRegister;    
  };

  class PrimitiveMapping : public IRelationalMapping {
    public:
    PrimitiveMapping( const Reflex::Type& attributeType, TableRegister& tableRegister  );
    ~PrimitiveMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class BlobMapping : public IRelationalMapping {
    public:
    BlobMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~BlobMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class OraReferenceMapping : public IRelationalMapping {
    public:
    OraReferenceMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~OraReferenceMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };

  class UniqueReferenceMapping : public IRelationalMapping {
    public:
    UniqueReferenceMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~UniqueReferenceMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };

  class OraPtrMapping : public IRelationalMapping {
    public:
    OraPtrMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~OraPtrMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };

  class ArrayMapping : public IRelationalMapping {
    public:
    ArrayMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~ArrayMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class CArrayMapping : public IRelationalMapping {
    public:
    CArrayMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~CArrayMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class ObjectMapping : public IRelationalMapping {
    public:
    ObjectMapping( const Reflex::Type& attributeType, TableRegister& tableRegister );

    ~ObjectMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    Reflex::Type m_type;
    TableRegister& m_tableRegister;
    
  };

  class EmptyMapping : public IRelationalMapping {
    public:
    EmptyMapping();

    ~EmptyMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
  };

}

#endif
