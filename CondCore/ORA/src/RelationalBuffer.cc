#include "RelationalBuffer.h"
#include "RelationalOperation.h"

ora::RelationalBuffer::RelationalBuffer( coral::ISchema& schema ):
  m_schema( schema ),
  m_operations(),
  m_volatileBuffers(),
  m_blobBuffer(){
}

ora::RelationalBuffer::~RelationalBuffer(){
  clear();
}

ora::InsertOperation& ora::RelationalBuffer::newInsert( const std::string& tableName ){
  InsertOperation* newOperation = new InsertOperation( tableName, m_schema );
  m_operations.push_back( newOperation );
  return *newOperation;
}

ora::BulkInsertOperation& ora::RelationalBuffer::newBulkInsert( const std::string& tableName ){
  BulkInsertOperation* newOperation = new BulkInsertOperation( tableName, m_schema );
  m_operations.push_back( newOperation );
  return *newOperation;  
}


ora::UpdateOperation& ora::RelationalBuffer::newUpdate( const std::string& tableName ){
  UpdateOperation* newOperation = new UpdateOperation( tableName, m_schema );
  m_operations.push_back( newOperation );
  return *newOperation;  
}

ora::DeleteOperation& ora::RelationalBuffer::newDelete( const std::string& tableName ){
  DeleteOperation* newOperation = new DeleteOperation( tableName, m_schema );
  m_operations.push_back( newOperation );
  return *newOperation;  
}


ora::RelationalBuffer& ora::RelationalBuffer::addVolatileBuffer(){
  RelationalBuffer* newBuffer = new RelationalBuffer( m_schema );
  m_volatileBuffers.push_back( newBuffer );
  return *newBuffer;
}

void ora::RelationalBuffer::storeBlob( boost::shared_ptr<coral::Blob> blob ){
  m_blobBuffer.push_back( blob );
}

void ora::RelationalBuffer::clear(){
  for( std::vector<IRelationalOperation*>::const_iterator iOp = m_operations.begin();
       iOp != m_operations.end(); ++iOp ){
    delete *iOp;
  }
  m_operations.clear();
  for( std::vector<RelationalBuffer*>::const_iterator iV = m_volatileBuffers.begin() ;
       iV != m_volatileBuffers.end(); ++iV ){
    delete *iV;
  }
  m_volatileBuffers.clear();
  m_blobBuffer.clear();
}

bool ora::RelationalBuffer::flush(){
  bool ok = true;
  std::vector<IRelationalOperation*>::const_iterator iOp = m_operations.begin();
  if( iOp != m_operations.end() ){
    ok = (*iOp)->execute();
    iOp++;
  }
  for( ; iOp != m_operations.end(); ++iOp ){
    if( ok ){
      (*iOp)->execute();
    } else {
      (*iOp)->reset();
    }
  }
  for( std::vector<RelationalBuffer*>::iterator iV = m_volatileBuffers.begin() ;
       iV != m_volatileBuffers.end(); ++iV ){
    (*iV)->flush();
    delete *iV;
  }
  m_volatileBuffers.clear();
  m_blobBuffer.clear();
  return ok;
}


