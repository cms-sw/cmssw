#ifndef INCLUDE_ORA_IDATABASESCHEMA_H
#define INCLUDE_ORA_IDATABASESCHEMA_H

//
#include <string> 
#include <map> 
#include <set> 
#include <vector>

namespace coral {
  class ISchema;
}

namespace ora {
  class IDatabaseTable {
    public:
    virtual ~IDatabaseTable(){}

    virtual bool exists() = 0;
    virtual void create() = 0;
    virtual void drop() = 0;
  };

  class IMainTable: public IDatabaseTable {
    public:
    static std::string schemaVersionParameterName();
    public:
    virtual ~IMainTable(){}
    virtual bool getParameters( std::map<std::string,std::string>& destination ) = 0;
  };

  class ISequenceTable : public IDatabaseTable{
    public:
    virtual ~ISequenceTable(){
    }

    virtual bool add( const std::string& sequenceName ) = 0;
    virtual bool getLastId( const std::string& sequenceName, int& lastId ) = 0;
    virtual void sinchronize( const std::string& sequenceName, int lastValue ) = 0;
    virtual void erase( const std::string& sequenceName ) = 0;
  };

  struct MappingRawElement {
      static std::string emptyScope();
      MappingRawElement();
      MappingRawElement(const MappingRawElement& rhs);
      MappingRawElement& operator==(const MappingRawElement& rhs);
      std::string scopeName;
      std::string variableName;
      std::string variableType;
      std::string elementType;
      std::string tableName;
      std::vector<std::string> columns;
  };

  struct MappingRawData {
      MappingRawData();
      explicit MappingRawData( const std::string& version );
      MappingRawElement& addElement( int elementId );
      std::string version;
      std::map< int, MappingRawElement> elements;
  };


  struct ContainerHeaderData {
      ContainerHeaderData( int contId,
                           const std::string& classN,
                           unsigned int numberObj );
      ContainerHeaderData( const ContainerHeaderData& rhs );
      ContainerHeaderData& operator=( const ContainerHeaderData& rhs );
      int id;
      std::string className;
      unsigned int numberOfObjects;
  };
  
  class IContainerHeaderTable: public IDatabaseTable  {
    public:
    virtual ~IContainerHeaderTable(){
    }
    virtual bool getContainerData( std::map<std::string, ContainerHeaderData>& destination ) = 0;
    virtual void addContainer( int id, const std::string& containerName, const std::string& className ) = 0;
    virtual void removeContainer( int id ) = 0;
    virtual void incrementNumberOfObjects( int id  ) = 0;
    virtual void decrementNumberOfObjects( int id  ) = 0;
    virtual void updateNumberOfObjects( const std::map<int,unsigned int>& numberOfObjectsForContainerIds ) = 0;
  };

  class IMappingSchema {
    public:
    virtual ~IMappingSchema(){
    }
    virtual bool getVersionList( std::set<std::string>& destination ) = 0;
    virtual bool getMapping( const std::string& version, MappingRawData& destination ) = 0;
    virtual void storeMapping( const MappingRawData& data ) = 0;
    virtual void removeMapping( const std::string& version ) = 0;
    virtual bool getContainerTableMap( std::map<std::string, int>& destination ) = 0;
    virtual bool getMappingVersionListForContainer( int containerId, std::set<std::string>& destination, bool onlyDependency=false ) = 0;
    virtual bool getDependentClassesInContainerMapping( int containerId, std::set<std::string>& destination ) = 0;
    virtual bool getClassVersionListForMappingVersion( const std::string& mappingVersion, std::set<std::string>& destination ) = 0;
    virtual bool getClassVersionListForContainer( int containerId, std::map<std::string,std::string>& versionMap ) = 0;
    virtual bool getMappingVersionListForTable( const std::string& tableName, std::set<std::string>& destination ) = 0;
    virtual bool selectMappingVersion( const std::string& classId, int containerId, std::string& destination ) = 0;
    virtual bool containerForMappingVersion( const std::string& mappingVersion, int& destination ) = 0;
    virtual void insertClassVersion( const std::string& className, const std::string& classVersion, const std::string& classId,
                                     int dependencyIndex, int containerId, const std::string& mappingVersion )= 0;
    virtual void setMappingVersion( const std::string& classId, int containerId, const std::string& mappingVersion ) = 0;
  };

  class INamingServiceTable: public IDatabaseTable  {
    public:
    virtual ~INamingServiceTable(){
    }
    virtual void setObjectName( const std::string& name, int contId, int itemId ) = 0;
    virtual bool getObjectByName( const std::string& name, std::pair<int,int>& destination ) = 0;
  };

  class IDatabaseSchema {
    public:

    static IDatabaseSchema* createSchemaHandle( coral::ISchema& schema );

    public:
    IDatabaseSchema( coral::ISchema& schema );
    virtual ~IDatabaseSchema(){
    }

    virtual bool exists() = 0;
    virtual void create() = 0;
    virtual void drop() = 0;

    virtual IMainTable& mainTable() = 0;
    virtual ISequenceTable& sequenceTable() = 0;
    virtual IDatabaseTable& mappingVersionTable() = 0;
    virtual IDatabaseTable& mappingElementTable() = 0;
    virtual IContainerHeaderTable& containerHeaderTable() = 0;
    virtual IDatabaseTable& classVersionTable() = 0;
    virtual IMappingSchema& mappingSchema() = 0;
    virtual INamingServiceTable& namingServiceTable() = 0;
    coral::ISchema& storageSchema();
    
    private:
    coral::ISchema& m_schema;
  };
  
}

#endif
  
    
    
