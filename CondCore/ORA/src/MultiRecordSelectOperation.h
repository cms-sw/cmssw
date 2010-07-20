#ifndef INCLUDE_ORA_MULTIRECORDSELECTOPERATION_H
#define INCLUDE_ORA_MULTIRECORDSELECTOPERATION_H

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
    void addId(const std::string& columnName);

    void addData(const std::string& columnName, const std::type_info& columnType );

    void addBlobData(const std::string& columnName);

    //void addMetadata( const std::string& columnName, const std::type_info& columnType );

    void addWhereId( const std::string& columnName );

    coral::AttributeList& data();
    coral::AttributeList& whereData();
    std::string& whereClause();
    
    private:
    SelectOperation m_query;
    std::vector<std::string> m_idCols;
    MultiIndexDataTrie m_cache;
    //coral::AttributeList* m_row;
    boost::shared_ptr<coral::AttributeList> m_row;
    
  };
  
}

#endif
