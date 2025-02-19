#ifndef INCLUDE_ORA_MULTIRECORDINSERTOPERATION_H
#define INCLUDE_ORA_MULTIRECORDINSERTOPERATION_H

#include "CondCore/ORA/interface/Record.h"
#include "RelationalOperation.h"

#define INSERTCACHESIZE 50000

namespace coral  {
  class AttibuteList;
}

namespace ora {

  class InsertCache {
    public:
    InsertCache( const RecordSpec& m_spec, const coral::AttributeList& data );
    ~InsertCache();
    void processNextIteration();
    const std::vector<Record*>& records() const;
    private:
    const RecordSpec& m_spec;
    std::vector<Record*> m_records;
    const coral::AttributeList& m_data;
  };

  class MultiRecordInsertOperation : public IRelationalData, public IRelationalOperation {
    public:
    MultiRecordInsertOperation( const std::string& tableName, coral::ISchema& schema );
    ~MultiRecordInsertOperation();
    InsertCache& setUp( int rowCacheSize );

    public:
    int addId( const std::string& columnName );
    int addData( const std::string& columnName, const std::type_info& columnType );
    int addBlobData(const std::string& columnName);
    int addWhereId( const std::string& columnName );
    coral::AttributeList& data();
    coral::AttributeList& whereData();
    std::string& whereClause();

    public:
    bool isRequired();
    bool execute();
    void reset();
    private:
    InputRelationalData m_relationalData;
    RecordSpec m_spec;
    std::string m_tableName;
    coral::ISchema& m_schema;
    std::vector<InsertCache*> m_bulkInserts;
  };
}

#endif
