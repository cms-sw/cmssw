#ifndef INCLUDE_ORA_ORADATABASESCHEMA_H
#define INCLUDE_ORA_ORADATABASESCHEMA_H

#include "IDatabaseSchema.h"
// externals
#include "CoralBase/AttributeList.h"

namespace ora {

  class OraMainTable: public IMainTable {
    public:
    static std::string version();
    static std::string tableName();
    static std::string parameterNameColumn();
    static std::string parameterValueColumn();
    public:
    explicit OraMainTable( coral::ISchema& dbSchema );
    virtual ~OraMainTable();
    bool getParameters( std::map<std::string,std::string>& destination );
    std::string schemaVersion();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };

  
  class OraSequenceTable : public ISequenceTable{
    public:
    static std::string tableName();
    static std::string sequenceNameColumn();
    static std::string sequenceValueColumn();    
    public:
    explicit OraSequenceTable( coral::ISchema& dbSchema );
    virtual ~OraSequenceTable();
    bool add( const std::string& sequenceName );
    bool getLastId( const std::string& sequenceName, int& lastId );
    void sinchronize( const std::string& sequenceName, int lastValue );
    void erase( const std::string& sequenceName );
    public:
    bool exists();
    void create();
    void drop();    
    private:
    coral::ISchema& m_schema;
  };

  class OraMappingVersionTable: public IDatabaseTable {
    public:
    static std::string tableName();
    static std::string mappingVersionColumn();
    public:
    explicit OraMappingVersionTable( coral::ISchema& dbSchema  );
    virtual ~OraMappingVersionTable();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };
  
  class OraMappingElementTable: public IDatabaseTable {
    public:
    static std::string tableName();
    static std::string mappingVersionColumn();
    static std::string elementIdColumn();
    static std::string elementTypeColumn();
    static std::string scopeNameColumn();
    static std::string variableNameColumn();
    static std::string variableParIndexColumn();
    static std::string variableTypeColumn();
    static std::string tableNameColumn();
    static std::string columnNameColumn();
    public:
    explicit OraMappingElementTable( coral::ISchema& dbSchema  );
    virtual ~OraMappingElementTable();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };

  class OraContainerHeaderTable: public IContainerHeaderTable {
    public:
    static std::string tableName();
    static std::string containerIdColumn();
    static std::string containerNameColumn();
    static std::string classNameColumn();
    static std::string numberOfObjectsColumn();
    public:
    explicit OraContainerHeaderTable( coral::ISchema& dbSchema );
    virtual ~OraContainerHeaderTable();
    bool getContainerData( std::map<std::string, ContainerHeaderData>& destination );
    void addContainer( int id, const std::string& containerName, const std::string& className );
    void removeContainer( int id );
    void incrementNumberOfObjects( int id  );
    void decrementNumberOfObjects( int id  );
    void updateNumberOfObjects( const std::map<int,unsigned int>& numberOfObjectsForContainerIds );
    public:
    //std::string name();
    bool exists();
    void create();
    void drop();
    private:
    void updateContainer( int id, const std::string& setClause );
    private:
    coral::ISchema& m_schema;
  };

  class OraClassVersionTable: public IDatabaseTable {
    public:
    static std::string tableName();
    static std::string classNameColumn();
    static std::string classVersionColumn();
    static std::string classIdColumn();
    static std::string dependencyIndexColumn();
    static std::string containerIdColumn();
    static std::string mappingVersionColumn();
    public:
    explicit OraClassVersionTable( coral::ISchema& dbSchema  );
    virtual ~OraClassVersionTable();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };

  class OraMappingSchema: public IMappingSchema {
    public:
    explicit OraMappingSchema( coral::ISchema& dbSchema  );
    virtual ~OraMappingSchema();
   public:
    bool getVersionList( std::set<std::string>& destination );
    bool getMapping( const std::string& version, MappingRawData& destination );
    void storeMapping( const MappingRawData& mapping );
    void removeMapping( const std::string& version );
    bool getContainerTableMap( std::map<std::string, int>& destination );
    bool getMappingVersionListForContainer( int containerId, std::set<std::string>& destination, bool onlyDependency=false );
    bool getDependentClassesInContainerMapping( int containerId, std::set<std::string>& destination );
    bool getClassVersionListForMappingVersion( const std::string& mappingVersion, std::set<std::string>& destination );
    bool getClassVersionListForContainer( int containerId, std::map<std::string,std::string>& versionMap );
    bool getMappingVersionListForTable( const std::string& tableName, std::set<std::string>& destination );
    bool selectMappingVersion( const std::string& classId, int containerId, std::string& destination );
    bool containerForMappingVersion( const std::string& mappingVersion, int& destination );
    void insertClassVersion( const std::string& className, const std::string& classVersion, const std::string& classId,
                             int dependencyIndex, int containerId, const std::string& mappingVersion );
    void setMappingVersion( const std::string& classId, int containerId, const std::string& mappingVersion );
    private:
    coral::ISchema& m_schema;
  };

  class OraDatabaseSchema: public IDatabaseSchema {
    public:
    static bool existsMainTable( coral::ISchema& dbSchema );
    
    public:
    explicit OraDatabaseSchema( coral::ISchema& dbSchema );
    virtual ~OraDatabaseSchema();
    
    bool exists();
    void create();
    void drop();

    IMainTable& mainTable();
    ISequenceTable& sequenceTable();
    IDatabaseTable& mappingVersionTable();
    IDatabaseTable& mappingElementTable();
    IContainerHeaderTable& containerHeaderTable();
    IDatabaseTable& classVersionTable();
    IMappingSchema& mappingSchema();

    private:
    coral::ISchema& m_schema;
    OraMainTable m_mainTable;
    OraSequenceTable m_sequenceTable;
    OraMappingVersionTable m_mappingVersionTable;
    OraMappingElementTable m_mappingElementTable;
    OraContainerHeaderTable m_containerHeaderTable;
    OraClassVersionTable m_classVersionTable;
    OraMappingSchema m_mappingSchema;
  };  
  
}

#endif
  
    
