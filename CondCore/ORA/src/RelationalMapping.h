#ifndef INCLUDE_ORA_RELATIONALMAPPING_H
#define INCLUDE_ORA_RELATIONALMAPPING_H

//
#include <utility>
#include <string>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {
  class TypeWithDict;
}

namespace ora {

  namespace RelationalMapping {
    size_t sizeInColumns(const edm::TypeWithDict& topLevelClassType );
    std::pair<bool,size_t> sizeInColumnsForCArray( const edm::TypeWithDict& arrayType );
    void _sizeInColumns(const edm::TypeWithDict& typ, size_t& sz, bool& hasDependencies );
    void _sizeInColumnsForCArray(const edm::TypeWithDict& typ, size_t& sz, bool& hasDependencies );
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
    IRelationalMapping* newProcessor( const edm::TypeWithDict& attributeType, bool blobStreaming=false );

    private:
    TableRegister& m_tableRegister;    
  };

  class PrimitiveMapping : public IRelationalMapping {
    public:
    PrimitiveMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister  );
    ~PrimitiveMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class BlobMapping : public IRelationalMapping {
    public:
    BlobMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~BlobMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class OraReferenceMapping : public IRelationalMapping {
    public:
    OraReferenceMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~OraReferenceMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };

  class UniqueReferenceMapping : public IRelationalMapping {
    public:
    UniqueReferenceMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~UniqueReferenceMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };

  class OraPtrMapping : public IRelationalMapping {
    public:
    OraPtrMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~OraPtrMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };

  class NamedRefMapping: public IRelationalMapping {
    public:
    NamedRefMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );
    ~NamedRefMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );

    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
  };

  class ArrayMapping : public IRelationalMapping {
    public:
    ArrayMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~ArrayMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class CArrayMapping : public IRelationalMapping {
    public:
    CArrayMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~CArrayMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
    TableRegister& m_tableRegister;
    
  };
  
  class ObjectMapping : public IRelationalMapping {
    public:
    ObjectMapping( const edm::TypeWithDict& attributeType, TableRegister& tableRegister );

    ~ObjectMapping();

    void process( MappingElement& parentElement, const std::string& attributeName,
                  const std::string& attributeNameForSchema, const std::string& scopeNameForSchema );
    private:
    edm::TypeWithDict m_type;
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
