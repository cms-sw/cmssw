#ifndef INCLUDE_ORA_RELATIONALBUFFER_H
#define INCLUDE_ORA_RELATIONALBUFFER_H

//
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
//
#include <memory>

namespace coral {
  class ISchema;
  class Blob;
}

namespace ora {

  class IRelationalOperation;
  class InsertOperation;
  class BulkInsertOperation;
  class MultiRecordInsertOperation;
  class UpdateOperation;
  class DeleteOperation;

  class RelationalBuffer {

    public:
    
    explicit RelationalBuffer( coral::ISchema& schema );

    virtual ~RelationalBuffer();

    InsertOperation& newInsert( const std::string& tableName );
    BulkInsertOperation& newBulkInsert( const std::string& tableName );
    MultiRecordInsertOperation& newMultiRecordInsert( const std::string& tableName );
    UpdateOperation& newUpdate( const std::string& tableName, bool addToResult=false );
    DeleteOperation& newDelete( const std::string& tableName, bool addToResult=false );

    RelationalBuffer& addVolatileBuffer();

    void storeBlob( boost::shared_ptr<coral::Blob> blob );

    void clear();
    bool flush();

    private:
    
    coral::ISchema& m_schema;
    std::vector< std::pair<IRelationalOperation*, bool> > m_operations;
    std::vector<RelationalBuffer*> m_volatileBuffers;
    std::vector< boost::shared_ptr<coral::Blob> > m_blobBuffer;
  };
  
}
#endif


  
