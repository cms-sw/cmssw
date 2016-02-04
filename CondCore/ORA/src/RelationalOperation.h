#ifndef INCLUDE_ORA_RELATIONALOPERATION_H
#define INCLUDE_ORA_RELATIONALOPERATION_H

// externals 
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeListSpecification.h"
//
#include <memory>

namespace coral  {
  class ISchema;
  class IBulkOperation;
  class IQuery;
  class ICursor;
}

namespace ora {

  class IRelationalOperation {
    public:
    
    virtual ~IRelationalOperation(){
    }

    virtual bool isRequired() = 0;

    virtual bool execute() = 0 ;

    virtual void reset() = 0;

  };
  
  typedef enum { Eq, Gt, Lt, Ge, Le } ConditionType;

  class IRelationalData {
    public:
    virtual ~IRelationalData(){
    }
    
    virtual int addId( const std::string& columnName ) = 0;

    virtual int addData( const std::string& columnName, const std::type_info& columnType ) = 0;

    virtual int addBlobData(const std::string& columnName) = 0;
    
    //virtual int addMetadata( const std::string& columnName, const std::type_info& columnType ) = 0;

    virtual int addWhereId( const std::string& columnName ) = 0;

    virtual coral::AttributeList& data() = 0;

    virtual coral::AttributeList& whereData() = 0;

    virtual std::string& whereClause() = 0;

};

  class InputRelationalData : public IRelationalData {

    public:

    InputRelationalData();

    virtual ~InputRelationalData();

    public:

    int addId(const std::string& columnName);

    int addData(const std::string& columnName, const std::type_info& columnType );

    int addBlobData(const std::string& columnName);

    int addWhereId( const std::string& columnName );

    coral::AttributeList& data();

    coral::AttributeList& whereData();

    std::string& whereClause();

    public:

    int addWhereId( const std::string& columnName, ConditionType cond );

    std::string& updateClause();

    private:

    coral::AttributeList m_data;
    std::string m_setClause;
    std::string m_whereClause;
  };
  
  class InsertOperation: public InputRelationalData, public IRelationalOperation {
    public:
    InsertOperation( const std::string& tableName, coral::ISchema& schema );
    ~InsertOperation();

    public:
    bool isRequired();
    bool execute();
    void reset();

    private:
    std::string m_tableName;
    coral::ISchema& m_schema;
  };
  
  class BulkInsertOperation : public InputRelationalData, public IRelationalOperation {
    public:
    BulkInsertOperation( const std::string& tableName, coral::ISchema& schema );
    ~BulkInsertOperation();
    coral::IBulkOperation& setUp( int rowCacheSize );

    public:
    bool isRequired();
    bool execute();
    void reset();
    private:
    std::string m_tableName;
    coral::ISchema& m_schema;
    std::vector<coral::IBulkOperation*> m_bulkOperations;
  };

  class UpdateOperation : public InputRelationalData, public IRelationalOperation {
      
    public:
    explicit UpdateOperation( const std::string& tableName, coral::ISchema& schema );
    ~UpdateOperation();

    public:
    bool isRequired();
    bool execute();
    void reset();
    private:
    std::string m_tableName;
    coral::ISchema& m_schema;

  };


  class DeleteOperation : public InputRelationalData, public IRelationalOperation {
    public:
    explicit DeleteOperation( const std::string& tableName, coral::ISchema& schema );
    ~DeleteOperation();

    public:
    bool isRequired();
    bool execute();
    void reset();
    private:
    std::string m_tableName;
    coral::ISchema& m_schema;

  };


  class SelectOperation : public IRelationalData {
    public:
    explicit SelectOperation( const std::string& tableName, coral::ISchema& schema );
    ~SelectOperation();

    void addOrderId(const std::string& columnName);
    bool nextCursorRow();
    void clear();
    void execute();
    coral::AttributeListSpecification& attributeListSpecification();

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
    coral::AttributeListSpecification* m_spec;
    coral::AttributeList m_whereData;
    std::string m_whereClause;
    std::vector<std::string> m_orderByCols;
    std::auto_ptr<coral::IQuery> m_query;
    coral::ICursor* m_cursor;
    std::string m_tableName;
    coral::ISchema& m_schema;
  };
  
}

#endif
