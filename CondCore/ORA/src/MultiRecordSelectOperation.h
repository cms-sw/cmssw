#ifndef INCLUDE_ORA_MULTIRECORDSELECTOPERATION_H
#define INCLUDE_ORA_MULTIRECORDSELECTOPERATION_H

#include "CondCore/ORA/interface/Record.h"
#include "RelationalOperation.h"
#include "MultiIndexDataTrie.h"

namespace ora {

  class MultiRecordSelectOperation : public IRelationalData {
    public:
    MultiRecordSelectOperation( const std::string& tableName, coral::ISchema& schema );
    ~MultiRecordSelectOperation();

    void addOrderId(const std::string& columnName); 
    void selectRow( const std::vector<int>& selection );
    size_t selectionSize( const std::vector<int>& selection, size_t numberOfIndexes );
    void clear();
    void execute();

    public:
    int addId(const std::string& columnName);

    int addData(const std::string& columnName, const std::type_info& columnType );

    int addBlobData(const std::string& columnName);

    //int addMetadata( const std::string& columnName, const std::type_info& columnType );

    int addWhereId( const std::string& columnName );

    coral::AttributeList& data();
    coral::AttributeList& whereData();
    std::string& whereClause();
    
    private:
    SelectOperation m_query;
    std::vector<std::string> m_idCols;
    MultiIndexDataTrie m_cache;
    RecordSpec m_spec;
    std::auto_ptr<coral::AttributeList> m_row;
    
  };
  
}

#endif
