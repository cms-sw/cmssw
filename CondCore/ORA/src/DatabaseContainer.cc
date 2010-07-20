#include "CondCore/ORA/interface/Exception.h"
#include "DatabaseContainer.h"
#include "DatabaseSession.h"
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

namespace ora {

  class WriteBuffer {
    public:
      explicit WriteBuffer( ContainerSchema& contSchema ):
        m_buffer(),
        m_operationBuffer( contSchema.storageSchema() ),
        m_topLevelElement(),
        m_writer(),
        m_topLevelInsert( 0 ){

        MappingElement& topLevelMapping = contSchema.mapping( true ).topElement();
        m_topLevelInsert = &m_operationBuffer.newInsert( topLevelMapping.tableName() );
        m_topLevelInsert->addId(  topLevelMapping.columnNames()[ 0 ] );
        const Reflex::Type& type = contSchema.type();
        MappingElement::iterator iMap = topLevelMapping.find( type.Name(Reflex::SCOPED) );
        // the first inner mapping is the relevant...
        if( iMap == topLevelMapping.end()){
          throwException("Could not find a mapping element for class \""+
                         type.Name(Reflex::SCOPED)+"\"",
                         "WriteBuffer::WriteBuffer");
        }
        MappingElement& mapping = iMap->second;
        RelationalStreamerFactory streamerFactory( contSchema );
        m_writer.reset( streamerFactory.newWriter( type, mapping ) );
        m_writer->build( m_topLevelElement, *m_topLevelInsert, m_operationBuffer );
      }
      

      ~WriteBuffer(){
      }

      void registerForWrite( int oid, const void* data ){
        m_buffer.push_back( std::make_pair(oid, data ) );
      }
      
      size_t flush(){
        size_t nobj = 0;
        for( std::vector<std::pair<int, const void*> >::const_iterator iW = m_buffer.begin();
             iW != m_buffer.end(); ++iW ){
          write( iW->first, iW->second );
          if( m_operationBuffer.flush() ) nobj++;
        }
        m_buffer.clear();
        return nobj;
      }

    private:
      void write( int oid, const void* data ){
        coral::AttributeList& dataBuff = m_topLevelInsert->data();
        dataBuff.begin()->data<int>() = oid;
        m_writer->write( oid, data );
      }      

    private:
      std::vector<std::pair<int, const void*> > m_buffer;
      RelationalBuffer m_operationBuffer;
      DataElement m_topLevelElement;
      std::auto_ptr<IRelationalWriter> m_writer;
      InsertOperation* m_topLevelInsert;
  };

  class UpdateBuffer {
    public:
      explicit UpdateBuffer( ContainerSchema& contSchema ):
        m_buffer(),
        m_operationBuffer( contSchema.storageSchema() ),
        m_topLevelElement(),
        m_depDeleter(),
        m_updater(),
        m_topLevelUpdate( 0 ){

        std::vector<MappingElement> dependentMappings;
        contSchema.mappingForDependentClasses( dependentMappings );
        m_depDeleter.reset( new RelationalDeleter( dependentMappings ) );
        m_depDeleter->build( m_operationBuffer );
        dependentMappings.clear();

        MappingElement& topLevelMapping = contSchema.mapping( true ).topElement();
        m_topLevelUpdate = &m_operationBuffer.newUpdate( topLevelMapping.tableName(), true );
        m_topLevelUpdate->addId(  topLevelMapping.columnNames()[ 0 ] );
        m_topLevelUpdate->addWhereId(  topLevelMapping.columnNames()[ 0 ] );
        const Reflex::Type& type = contSchema.type();
        MappingElement::iterator iMap = topLevelMapping.find( type.Name(Reflex::SCOPED) );
        // the first inner mapping is the relevant...
        if( iMap == topLevelMapping.end()){
          throwException("Could not find a mapping element for class \""+
                         type.Name(Reflex::SCOPED)+"\"",
                         "UpdateBuffer::UpdateBuffer");
        }
        MappingElement& mapping = iMap->second;
        RelationalStreamerFactory streamerFactory( contSchema );
        m_updater.reset( streamerFactory.newUpdater( type, mapping ));
        m_updater->build( m_topLevelElement, *m_topLevelUpdate, m_operationBuffer );
      }
      
      ~UpdateBuffer(){
      }
      

      void registerForUpdate( int oid, const void* data ){
        m_buffer.push_back( std::make_pair( oid, data ));
      }
      

      size_t flush(){
        size_t nobj = 0;
        for( std::vector<std::pair<int, const void*> >::const_iterator iU = m_buffer.begin();
             iU != m_buffer.end(); ++iU ){
          update( iU->first, iU->second );
          if( m_operationBuffer.flush()) nobj++;
        }
        m_buffer.clear();
        return nobj;
      }
      

    private:
      void update( int oid, const void* data ){
        // erase the dependencies (cannot be updated...)
        m_depDeleter->erase( oid );
        coral::AttributeList& dataBuff = m_topLevelUpdate->data();
        dataBuff.begin()->data<int>() = oid;
        coral::AttributeList& whereDataBuff = m_topLevelUpdate->whereData();
        whereDataBuff.begin()->data<int>() = oid;
        m_updater->update( oid, data );
      }
      

    private:
      std::vector<std::pair<int, const void*> > m_buffer;
      RelationalBuffer m_operationBuffer;
      DataElement m_topLevelElement;
      std::auto_ptr<RelationalDeleter> m_depDeleter;
      std::auto_ptr<IRelationalUpdater> m_updater;
      UpdateOperation* m_topLevelUpdate;
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
        MappingElement::iterator iMap = topLevelMapping.find( m_type.Name(Reflex::SCOPED) );
        // the first inner mapping is the good one ...
        if( iMap == topLevelMapping.end()){
          throwException("Could not find a mapping element for class \""+
                         m_type.Name(Reflex::SCOPED)+"\"",
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
        }
        m_reader->clear();
        m_topLevelQuery.clear();
        return destination;
      }

      const Reflex::Type& type(){
        return m_type;
      }
      
    private:
      DataElement m_topLevelElement;
      const Reflex::Type& m_type;
      std::auto_ptr<IRelationalReader> m_reader;
      SelectOperation m_topLevelQuery;
  };

  class DeleteBuffer {
    public:
      explicit DeleteBuffer( ContainerSchema& contSchema ):
        m_buffer(),
        m_operationBuffer( contSchema.storageSchema() ),
        m_mainDeleter(),
        m_depDeleter(){
        m_mainDeleter.reset( new RelationalDeleter( contSchema.mapping().topElement() ));
        m_mainDeleter->build( m_operationBuffer );
        
        std::vector<MappingElement> dependentMappings;
        contSchema.mappingForDependentClasses( dependentMappings );
        m_depDeleter.reset( new RelationalDeleter( dependentMappings ) );
        m_depDeleter->build( m_operationBuffer );
        dependentMappings.clear();                             
      }
      
      ~DeleteBuffer(){
      }
      

      void registerForDelete( int oid ){
        m_buffer.push_back( oid );
      }

      size_t flush(){
        size_t nobj = 0;
        for( std::vector<int>::const_iterator iD = m_buffer.begin();
             iD != m_buffer.end(); ++iD ){
          m_depDeleter->erase( *iD );
          m_mainDeleter->erase( *iD );
          if( m_operationBuffer.flush() ) nobj++;
        }
        m_buffer.clear();
        return nobj;
      }
      
    private:
      std::vector<int> m_buffer;
      RelationalBuffer m_operationBuffer;
      std::auto_ptr<RelationalDeleter> m_mainDeleter;
      std::auto_ptr<RelationalDeleter> m_depDeleter;
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

void* ora::IteratorBuffer::getItemAsType( const Reflex::Type& asType ){
  if( !ClassUtils::isType( type(), asType ) ){
    throwException("Provided output object type \""+asType.Name(Reflex::SCOPED)+"\" does not match with the container type \""+
                   type().Name(Reflex::SCOPED)+"\"","IteratorBuffer::getItemAsType");
  } 
  return getItem();
}

int ora::IteratorBuffer::itemId(){
  return m_itemId;
}

const Reflex::Type& ora::IteratorBuffer::type(){
  return m_readBuffer.type();
}
      
ora::DatabaseContainer::DatabaseContainer( int contId,
                                           const std::string& containerName,
                                           const std::string& className,
                                           unsigned int containerSize,
                                           DatabaseSession& session ):
  m_schema( new ContainerSchema(contId, containerName, className, session) ),
  m_writeBuffer(),
  m_updateBuffer(),
  m_readBuffer(),
  m_deleteBuffer(),
  m_iteratorBuffer(),
  m_size( containerSize ),
  m_containerUpdateTable( session.containerUpdateTable() ){
}

ora::DatabaseContainer::DatabaseContainer( int contId,
                                           const std::string& containerName,
                                           const Reflex::Type& containerType,
                                           DatabaseSession& session ):
  m_schema( new ContainerSchema(contId, containerName, containerType, session) ),
  m_writeBuffer(),
  m_updateBuffer(),
  m_readBuffer(),
  m_deleteBuffer(),
  m_iteratorBuffer(),
  m_size(0),
  m_containerUpdateTable( session.containerUpdateTable() ){
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

const Reflex::Type& ora::DatabaseContainer::type(){
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

void ora::DatabaseContainer::create(){
  m_schema->create();
}

void ora::DatabaseContainer::drop(){
  m_schema->drop();
}

void ora::DatabaseContainer::extendSchema( const Reflex::Type& dependentType ){
  m_schema->extendIfRequired( dependentType );
}

void* ora::DatabaseContainer::fetchItem(int itemId){

  if(!m_readBuffer.get()){
    m_readBuffer.reset( new ReadBuffer( *m_schema ) );
  }
  return m_readBuffer->read( itemId );
}

void* ora::DatabaseContainer::fetchItemAsType(int itemId,
                                              const Reflex::Type& asType){
  if(!m_readBuffer.get()){
    m_readBuffer.reset( new ReadBuffer( *m_schema ) );
  }
  if( !ClassUtils::isType( type(), asType ) ){
    throwException("Provided output object type \""+asType.Name(Reflex::SCOPED)+"\" does not match with the container type \""+
                   type().Name(Reflex::SCOPED)+"\"","DatabaseContainer::fetchItemAsType");
  } 
  return m_readBuffer->read( itemId );
}

int ora::DatabaseContainer::insertItem( const void* data,
                                        const Reflex::Type& dataType ){
  if(!m_writeBuffer.get()){
    m_writeBuffer.reset( new WriteBuffer( *m_schema ) );
  }
  Reflex::Type inputResType = ClassUtils::resolvedType( dataType );
  Reflex::Type contType = ClassUtils::resolvedType(m_schema->type());
  if( inputResType.Name()!= contType.Name() && !inputResType.HasBase( contType ) ){
    throwException( "Provided input object type=\""+inputResType.Name()+
                    "\" does not match with the container type=\""+contType.Name()+"\"",
                    "DatabaseContainer::insertItem" );
  }

  int newId = m_schema->containerSequences().getNextId( MappingRules::sequenceNameForContainer( m_schema->containerName()) );
  m_writeBuffer->registerForWrite( newId, data );
  return newId;
}

void ora::DatabaseContainer::updateItem( int itemId,
                                         const void* data,
                                         const Reflex::Type& dataType ){
  if(!m_updateBuffer.get()){
    m_updateBuffer.reset( new UpdateBuffer( *m_schema ) );
  }
  Reflex::Type inputResType = ClassUtils::resolvedType( dataType );
  Reflex::Type contType = ClassUtils::resolvedType(m_schema->type());
  if( inputResType.Name()!= contType.Name() && !inputResType.HasBase( contType ) ){
    throwException( "Provided input object type=\""+inputResType.Name()+"\" does not match with the container type=\""+
                    contType.Name()+"\".",
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



