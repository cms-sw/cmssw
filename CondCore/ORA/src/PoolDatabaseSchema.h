#ifndef INCLUDE_ORA_POOLDATABASESCHEMA_H
#define INCLUDE_ORA_POOLDATABASESCHEMA_H

#include "IDatabaseSchema.h"
// externals
#include "CoralBase/AttributeList.h"

namespace ora {

    struct PoolDbCacheData {
      PoolDbCacheData();
        
      PoolDbCacheData( int id, const std::string& name, const std::string& mappingVersion, unsigned int nobjWritten );

      ~PoolDbCacheData();

      PoolDbCacheData( const PoolDbCacheData& rhs );

      PoolDbCacheData& operator=( const PoolDbCacheData& rhs );

      int m_id;
      std::string m_name;
      std::string m_mappingVersion;
      unsigned int m_nobjWr;
  };
  
  class PoolDbCache {
    public:
    PoolDbCache();
    ~PoolDbCache();
    void add( int id, const PoolDbCacheData& data );
    const std::string& nameById( int id );
    PoolDbCacheData& find( int id );
    void remove( int id );
    std::map<std::string,PoolDbCacheData* >& sequences();
    void clear();
    
    private:
    PoolDbCacheData m_databaseData;
    PoolDbCacheData m_mappingData;
    std::map<int,PoolDbCacheData > m_idMap;
    std::map<std::string,PoolDbCacheData* > m_sequences;
  };

  class PoolMainTable: public IMainTable {
    public:
    static std::string schemaVersion();
    static std::string tableName();
    public:
    explicit PoolMainTable( coral::ISchema& dbSchema );
    virtual ~PoolMainTable();
    bool getParameters( std::map<std::string,std::string>& destination );
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };

  
  class PoolSequenceTable : public ISequenceTable{
    public:
    static std::string tableName();
    static std::string sequenceNameColumn();
    static std::string sequenceValueColumn();    
    public:
    explicit PoolSequenceTable( coral::ISchema& dbSchema );
    virtual ~PoolSequenceTable();
    void init( PoolDbCache& dbCache );
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
    PoolDbCache* m_dbCache;
  };

  class PoolMappingVersionTable: public IDatabaseTable {
    public:
    static std::string tableName();
    static std::string mappingVersionColumn();
    static std::string containerNameColumn();
    public:
    explicit PoolMappingVersionTable( coral::ISchema& dbSchema  );
    virtual ~PoolMappingVersionTable();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };
  
  class PoolMappingElementTable: public IDatabaseTable {
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
    explicit PoolMappingElementTable( coral::ISchema& dbSchema  );
    virtual ~PoolMappingElementTable();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };
  
  class PoolContainerHeaderTable: public IContainerHeaderTable {
    public:
    static std::string tableName();
    static std::string containerIdColumn();
    static std::string containerNameColumn();
    static std::string containerTypeColumn();
    static std::string tableNameColumn();
    static std::string classNameColumn();
    static std::string baseMappingVersionColumn();
    static std::string numberOfWrittenObjectsColumn();
    static std::string numberOfDeletedObjectsColumn();
    static std::string homogeneousContainerType();
    public:
    explicit PoolContainerHeaderTable( coral::ISchema& dbSchema );
    virtual ~PoolContainerHeaderTable();
    void init( PoolDbCache& dbCache );
    bool getContainerData( std::map<std::string, ContainerHeaderData>& destination );
    void addContainer( int id, const std::string& containerName, const std::string& className );
    void removeContainer( int id );
    void incrementNumberOfObjects( int id  );
    void decrementNumberOfObjects( int id  );
    void updateNumberOfObjects( const std::map<int,unsigned int>& numberOfObjectsForContainerIds );
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
    PoolDbCache* m_dbCache;
  };

  class PoolClassVersionTable: public IDatabaseTable {
    public:
    static std::string tableName();
    static std::string classVersionColumn();
    static std::string containerNameColumn();
    static std::string mappingVersionColumn();
    public:
    explicit PoolClassVersionTable( coral::ISchema& dbSchema  );
    virtual ~PoolClassVersionTable();
    public:
    bool exists();
    void create();
    void drop();
    private:
    coral::ISchema& m_schema;
  };

  class PoolMappingSchema: public IMappingSchema {
    public:
    static std::string emptyScope();
    public:
    explicit PoolMappingSchema( coral::ISchema& dbSchema );
    virtual ~PoolMappingSchema();
    void init( PoolDbCache& dbCache );
    public:
    bool getVersionList( std::set<std::string>& destination );
    bool getMapping( const std::string& version, MappingRawData& destination );
    void storeMapping( const MappingRawData& mapping );
    void removeMapping( const std::string& version );
    bool getContainerTableMap( std::map<std::string, int>& destination );
    bool getMappingVersionListForContainer( int containerId, std::set<std::string>& destination, bool onlyDependency=false );
    bool getMappingVersionListForTable( const std::string& tableName, std::set<std::string>& destination );
    bool selectMappingVersion( const std::string& classId, int containerId, std::string& destination );
    bool containerForMappingVersion( const std::string& mappingVersion, int& destination );
    void insertClassVersion( const std::string& className, const std::string& classVersion, const std::string& classId,
                             int dependencyIndex, int containerId, const std::string& mappingVersion );
    void setMappingVersion( const std::string& classId, int containerId, const std::string& mappingVersion );
    private:
    coral::ISchema& m_schema;
    PoolDbCache* m_dbCache;
  };
  
  class PoolDatabaseSchema: public IDatabaseSchema {
    public:
    static bool existsMainTable( coral::ISchema& dbSchema );
    public:
    explicit PoolDatabaseSchema( coral::ISchema& dbSchema );
    virtual ~PoolDatabaseSchema();

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
    PoolDbCache m_dbCache;
    PoolMainTable m_mainTable;
    PoolSequenceTable m_sequenceTable;
    PoolMappingVersionTable m_mappingVersionTable;
    PoolMappingElementTable m_mappingElementTable;
    PoolContainerHeaderTable m_containerHeaderTable;
    PoolClassVersionTable m_classVersionTable;
    PoolMappingSchema m_mappingSchema;
  };  
  
}

#endif
  
    
