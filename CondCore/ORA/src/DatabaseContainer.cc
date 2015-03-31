#include "CondCore/ORA/interface/Exception.h"
#include "DatabaseContainer.h"
#include "DatabaseSession.h"
#include "IDatabaseSchema.h"
#include "ContainerSchema.h"
#include "IRelationalStreamer.h"
#include "RelationalBuffer.h"
#include "RelationalOperation.h"
#include "DataElement.h"
#include "RelationalDeleter.h"
#include "RelationalStreamerFactory.h"
#include "IDatabaseSchema.h"
#include "ClassUtils.h"
#include "MappingRules.h"
// externals
#include "CoralBase/Attribute.h"
//
#include <boost/lexical_cast.hpp>

namespace ora {

  class WriteBuffer {
    public:
      explicit WriteBuffer( ContainerSchema& contSchema ):
        m_buffer(),
        m_contSchema( contSchema ){
      }

      ~WriteBuffer(){
      }

      void registerForWrite( int oid, const void* data ){
        m_buffer.push_back( std::make_pair(oid, data ) );
      }
      
      size_t flush(){
        size_t nobj = 0;
        if( m_buffer.size() ){
	  MappingElement& topLevelMapping = m_contSchema.mapping( true ).topElement();
	  RelationalBuffer operationBuffer( m_contSchema.storageSchema() );
	  InsertOperation* topLevelInsert = &operationBuffer.newInsert( topLevelMapping.tableName() );
	  topLevelInsert->addId(  topLevelMapping.columnNames()[ 0 ] );
	  const edm::TypeWithDict& type = m_contSchema.type();
	  MappingElement::iterator iMap = topLevelMapping.find( type.cppName() );
	  // the first inner mapping is the relevant...
	  if( iMap == topLevelMapping.end()){
	    throwException("Could not find a mapping element for class \""+
			   type.cppName()+"\"",
			   "WriteBuffer::flush");
	  }
	  MappingElement& mapping = iMap->second;
	  RelationalStreamerFactory streamerFactory( m_contSchema );
	  DataElement topLevelElement;
	  std::auto_ptr<IRelationalWriter> writer( streamerFactory.newWriter( type, mapping ) );
	  writer->build( topLevelElement, *topLevelInsert, operationBuffer );
	  
	  for( std::vector<std::pair<int, const void*> >::const_iterator iW = m_buffer.begin();
	       iW != m_buffer.end(); ++iW ){
	    int oid = iW->first;
	    const void* data = iW->second;
	    coral::AttributeList& dataBuff = topLevelInsert->data();
	    dataBuff.begin()->data<int>() = oid;
	    writer->write( oid, data );
	    if( operationBuffer.flush() ) nobj++;
	  }
	  m_buffer.clear();
	}
        return nobj;
      }


  private:
    std::vector<std::pair<int, const void*> > m_buffer;
    ContainerSchema& m_contSchema;

  };

  class UpdateBuffer {
    public:
      explicit UpdateBuffer( ContainerSchema& contSchema ):
        m_buffer(),
        m_contSchema( contSchema ){
      }
      
      ~UpdateBuffer(){
      }
      

      void registerForUpdate( int oid, const void* data ){
        m_buffer.push_back( std::make_pair( oid, data ));
      }
      

      size_t flush(){
        size_t nobj = 0;
	if( m_buffer.size() ){
	  RelationalBuffer operationBuffer( m_contSchema.storageSchema() );
	  std::vector<MappingElement> dependentMappings;
	  m_contSchema.mappingForDependentClasses( dependentMappings );
	  RelationalDeleter depDeleter( dependentMappings );
	  depDeleter.build( operationBuffer );
	  dependentMappings.clear();
	  
	  MappingElement& topLevelMapping = m_contSchema.mapping( true ).topElement();
	  UpdateOperation* topLevelUpdate = &operationBuffer.newUpdate( topLevelMapping.tableName(), true );
	  topLevelUpdate->addId(  topLevelMapping.columnNames()[ 0 ] );
	  topLevelUpdate->addWhereId(  topLevelMapping.columnNames()[ 0 ] );
	  const edm::TypeWithDict& type = m_contSchema.type();
	  MappingElement::iterator iMap = topLevelMapping.find( type.cppName() );
	  // the first inner mapping is the relevant...
	  if( iMap == topLevelMapping.end()){
	    throwException("Could not find a mapping element for class \""+
			   type.cppName()+"\"",
			   "UpdateBuffer::flush");
	  }
	  MappingElement& mapping = iMap->second;
	  RelationalStreamerFactory streamerFactory( m_contSchema );
	  DataElement topLevelElement;
	  std::auto_ptr<IRelationalUpdater> updater( streamerFactory.newUpdater( type, mapping ));
	  updater->build( topLevelElement, *topLevelUpdate, operationBuffer );
	  for( std::vector<std::pair<int, const void*> >::const_iterator iU = m_buffer.begin();
	       iU != m_buffer.end(); ++iU ){
	    int oid = iU->first;
	    const void* data = iU->second;
	    // erase the dependencies (they cannot be updated...)
	    depDeleter.erase( oid );
	    coral::AttributeList& dataBuff = topLevelUpdate->data();
	    dataBuff.begin()->data<int>() = oid;
	    coral::AttributeList& whereDataBuff = topLevelUpdate->whereData();
	    whereDataBuff.begin()->data<int>() = oid;
	    updater->update( oid, data );
	    if( operationBuffer.flush()) nobj++;
	  }
	  m_buffer.clear();
	}
        return nobj;
      }
      
    private:
    std::vector<std::pair<int, const void*> > m_buffer;
    ContainerSchema& m_contSchema;
  };

  class ReadBuffer {
    public:
      explicit ReadBuffer( ContainerSchema& contSchema ):
        m_topLevelElement(),
        m_type( contSchema.type() ),
        m_reader(),
        m_topLevelQuery( contSchema.mapping().topElement().tableName(), contSchema.storageSchema() ){

        MappingElement& topLevelMapping = contSchema.mapping().topElement();
        m_topLevelQuery.addWhereId(  topLevelMapping.columnNames()[ 0 ] );
        MappingElement::iterator iMap = topLevelMapping.find( m_type.cppName() );
        // the first inner mapping is the good one ...
        if( iMap == topLevelMapping.end()){
          throwException("Could not find a mapping element for class \""+
                         m_type.cppName()+"\"",
                         "ReadBuffer::ReadBuffer");
        }
        MappingElement& mapping = iMap->second;
        RelationalStreamerFactory streamerFactory( contSchema );
        m_reader.reset( streamerFactory.newReader( m_type, mapping )) ;
        m_reader->build( m_topLevelElement , m_topLevelQuery );
      }

      ~ReadBuffer(){
      }
      
      void* read( int oid ){
        coral::AttributeList& dataBuff = m_topLevelQuery.whereData();
        dataBuff.begin()->data<int>() = oid;
        m_topLevelQuery.execute();
        m_reader->select( oid );
        void* destination = 0;
        if( m_topLevelQuery.nextCursorRow() ){
          destination = ClassUtils::constructObject( m_type );
          m_reader->read( destination );
        } else {
	  throwException("Object with item id "+boost::lexical_cast<std::string>(oid)+" has not been found in the database.",
			 "ReadBuffer::read");
	}
        m_reader->clear();
        m_topLevelQuery.clear();
        return destination;
      }

      const edm::TypeWithDict& type(){
        return m_type;
      }
      
    private:
      DataElement m_topLevelElement;
      const edm::TypeWithDict& m_type;
      std::auto_ptr<IRelationalReader> m_reader;
      SelectOperation m_topLevelQuery;
  };

  class DeleteBuffer {
    public:
      explicit DeleteBuffer( ContainerSchema& contSchema ):
        m_buffer(),
        m_contSchema( contSchema ){
      }
      
      ~DeleteBuffer(){
      }
      

      void registerForDelete( int oid ){
        m_buffer.push_back( oid );
      }

      size_t flush(){
        size_t nobj = 0;
	if( m_buffer.size()) {
	  RelationalBuffer operationBuffer( m_contSchema.storageSchema() );
	  RelationalDeleter mainDeleter( m_contSchema.mapping().topElement() );
	  mainDeleter.build( operationBuffer );
	  std::vector<MappingElement> dependentMappings;
	  m_contSchema.mappingForDependentClasses( dependentMappings );
	  RelationalDeleter depDeleter(  dependentMappings );
	  depDeleter.build( operationBuffer );
	  dependentMappings.clear();                             
	  
	  for( std::vector<int>::const_iterator iD = m_buffer.begin();
	       iD != m_buffer.end(); ++iD ){
	    depDeleter.erase( *iD );
	    mainDeleter.erase( *iD );
	    if( operationBuffer.flush() ) nobj++;
	  }
	  m_buffer.clear();
	}
        return nobj;
      }
      
    private:
    std::vector<int> m_buffer;
    ContainerSchema& m_contSchema;
  };
  
}

ora::IteratorBuffer::IteratorBuffer( ContainerSchema& schema,
                                     ReadBuffer& buffer ):
  m_query( schema.mapping().topElement().tableName(), schema.storageSchema() ),
  m_itemId( -1 ),
  m_readBuffer( buffer ){
  const std::string& idCol = schema.mapping().topElement().columnNames()[0];
  m_query.addId( idCol );
  m_query.addOrderId( idCol );
}

ora::IteratorBuffer::~IteratorBuffer(){
}

void ora::IteratorBuffer::reset(){
  m_query.execute();
}

bool ora::IteratorBuffer::next(){
  bool prevValid = (m_itemId != -1);
  bool currValid = false;
  m_itemId = -1;
  if( m_query.nextCursorRow() ){
    coral::AttributeList& row = m_query.data();
    m_itemId = row.begin()->data<int>();
    currValid = true;
  }
  
  if( !currValid && prevValid ) m_query.clear();
  return currValid;
}

void* ora::IteratorBuffer::getItem(){
  void* ret = 0;
  if( m_itemId != -1 ){
    ret =  m_readBuffer.read( m_itemId );
  }
  return ret;
}

void* ora::IteratorBuffer::getItemAsType( const edm::TypeWithDict& asType ){
  if( !ClassUtils::isType( type(), asType ) ){
    throwException("Provided output object type \""+asType.cppName()+"\" does not match with the container type \""+type().cppName(), 
		   "ora::IteratorBuffer::getItemsAsType" );
  } 
  return getItem();
}

int ora::IteratorBuffer::itemId(){
  return m_itemId;
}

const edm::TypeWithDict& ora::IteratorBuffer::type(){
  return m_readBuffer.type();
}
      
ora::DatabaseContainer::DatabaseContainer( int contId,
                                           const std::string& containerName,
                                           const std::string& className,
                                           unsigned int containerSize,
                                           DatabaseSession& session ):
  m_dbSchema( session.schema() ),
  m_schema( new ContainerSchema(contId, containerName, className, session) ),
  m_writeBuffer(),
  m_updateBuffer(),
  m_readBuffer(),
  m_deleteBuffer(),
  m_iteratorBuffer(),
  m_size( containerSize ),
  m_containerUpdateTable( session.containerUpdateTable() ),
  m_lock( false ){
}

ora::DatabaseContainer::DatabaseContainer( int contId,
                                           const std::string& containerName,
                                           const edm::TypeWithDict& containerType,
                                           DatabaseSession& session ):
  m_dbSchema( session.schema() ),
  m_schema( new ContainerSchema(contId, containerName, containerType, session) ),
  m_writeBuffer(),
  m_updateBuffer(),
  m_readBuffer(),
  m_deleteBuffer(),
  m_iteratorBuffer(),
  m_size(0),
  m_containerUpdateTable( session.containerUpdateTable() ),
  m_lock( false ) {
}

ora::DatabaseContainer::~DatabaseContainer(){
  m_iteratorBuffer.clear();
}

int ora::DatabaseContainer::id(){
  return m_schema->containerId();
}

const std::string& ora::DatabaseContainer::name(){
  return m_schema->containerName();
}

const std::string& ora::DatabaseContainer::className(){
  return m_schema->className();
}

const edm::TypeWithDict& ora::DatabaseContainer::type(){
  return m_schema->type();
}

const std::string& ora::DatabaseContainer::mappingVersion(){
  return m_schema->mappingVersion();
}

size_t ora::DatabaseContainer::size(){
  return m_size;
}

ora::Handle<ora::IteratorBuffer> ora::DatabaseContainer::iteratorBuffer(){
  if(!m_readBuffer.get()){
    m_readBuffer.reset( new ReadBuffer( *m_schema ) );
  }
  if( !m_iteratorBuffer ){
    m_iteratorBuffer.reset( new IteratorBuffer(*m_schema, *m_readBuffer ) );
    m_iteratorBuffer->reset();
  }
  return m_iteratorBuffer;
}

bool ora::DatabaseContainer::lock(){
  if( !m_lock ){
    ContainerHeaderData headerData;
    m_lock = m_dbSchema.containerHeaderTable().lockContainer( m_schema->containerId(), headerData );
    if(!m_lock) throwException("Container \""+name()+"\" has been dropped.","DatabaseContainer::lock()");
    // once the lock has been taken over, update the size in case has been changed...
    m_size = headerData.numberOfObjects;
  }
  return m_lock;
}

bool ora::DatabaseContainer::isLocked(){
  return m_lock;
}

void ora::DatabaseContainer::create(){
  m_schema->create();
}

void ora::DatabaseContainer::drop(){
  if(!m_schema->dbSession().testDropPermission()){
    throwException("Drop permission has been denied for the current user.",
		   "DatabaseContainer::drop");
  }
  m_schema->drop();
  m_containerUpdateTable.remove( m_schema->containerId() );
}

void ora::DatabaseContainer::extendSchema( const edm::TypeWithDict& dependentType ){
  m_schema->extendIfRequired( dependentType );
}

void ora::DatabaseContainer::setAccessPermission( const std::string& principal, 
						  bool forWrite ){
  m_schema->setAccessPermission( principal, forWrite );
}

void* ora::DatabaseContainer::fetchItem(int itemId){

  if(!m_readBuffer.get()){
    m_readBuffer.reset( new ReadBuffer( *m_schema ) );
  }
  return m_readBuffer->read( itemId );
}

void* ora::DatabaseContainer::fetchItemAsType(int itemId,
                                              const edm::TypeWithDict& asType){
  if(!m_readBuffer.get()){
    m_readBuffer.reset( new ReadBuffer( *m_schema ) );
  }
  if( !ClassUtils::isType( type(), asType ) ){
    throwException("Provided output object type \""+asType.cppName()+"\" does not match with the container type \""+type().cppName(), 
		   "ora::DatabaseContainer::fetchItemAsType" );
  } 
  return m_readBuffer->read( itemId );
}

int ora::DatabaseContainer::insertItem( const void* data,
                                        const edm::TypeWithDict& dataType ){
  if(!m_writeBuffer.get()){
    m_writeBuffer.reset( new WriteBuffer( *m_schema ) );
  }
  edm::TypeWithDict inputResType = ClassUtils::resolvedType( dataType );
  edm::TypeWithDict contType = ClassUtils::resolvedType(m_schema->type());
  if( inputResType.name()!= contType.name() && !inputResType.hasBase( contType ) ){
    throwException( "Provided input object type=\""+inputResType.name()+
                    "\" does not match with the container type=\""+contType.name()+"\"",
                    "DatabaseContainer::insertItem" );
  }

  int newId = m_schema->containerSequences().getNextId( MappingRules::sequenceNameForContainer( m_schema->containerName()) );
  m_writeBuffer->registerForWrite( newId, data );
  return newId;
}

void ora::DatabaseContainer::updateItem( int itemId,
                                         const void* data,
                                         const edm::TypeWithDict& dataType ){
  if(!m_updateBuffer.get()){
    m_updateBuffer.reset( new UpdateBuffer( *m_schema ) );
  }
  edm::TypeWithDict inputResType = ClassUtils::resolvedType( dataType );
  edm::TypeWithDict contType = ClassUtils::resolvedType(m_schema->type());
  if( inputResType.name()!= contType.name() && !inputResType.hasBase( contType ) ){
    throwException( "Provided input object type=\""+inputResType.name()+"\" does not match with the container type=\""+
                    contType.name()+"\".",
                    "DatabaseContainer::updateItem" );
  }

  m_updateBuffer->registerForUpdate( itemId, data );
}

void ora::DatabaseContainer::erase( int itemId ){
  if(!m_deleteBuffer.get()){
    m_deleteBuffer.reset( new DeleteBuffer( *m_schema ) );
  }
  m_deleteBuffer->registerForDelete( itemId );
}

void ora::DatabaseContainer::flush(){
  size_t prevSize = m_size;
  if(m_writeBuffer.get()) m_size += m_writeBuffer->flush(); 
  if(m_updateBuffer.get()) m_updateBuffer->flush();
  if(m_deleteBuffer.get()) m_size -= m_deleteBuffer->flush();
  m_schema->containerSequences().sinchronizeAll();
  if( prevSize != m_size ){
    m_containerUpdateTable.takeNote( id(), m_size );
  }
}

void ora::DatabaseContainer::setItemName( const std::string& name, 
                                          int itemId ){
  m_schema->dbSession().setObjectName( name, m_schema->containerId(), itemId );
}

bool ora::DatabaseContainer::getNames( std::vector<std::string>& destination ){
  return m_schema->dbSession().getNamesForContainer( m_schema->containerId(), destination );
}


