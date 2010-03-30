#include "CondCore/ORA/interface/Exception.h"
#include "STLContainerStreamer.h"
#include "ClassUtils.h"
#include "MappingElement.h"
#include "DataElement.h"
#include "ContainerSchema.h"
#include "RelationalBuffer.h"
#include "RelationalOperation.h"
#include "MultiRecordSelectOperation.h"
#include "RelationalStreamerFactory.h"
#include "ArrayHandlerFactory.h"
#include "IArrayHandler.h"
#include "ArrayCommonImpl.h"
// externals
#include "CoralBase/Attribute.h"
#include "RelationalAccess/IBulkOperation.h"
#include "Reflex/Member.h"

ora::STLContainerWriter::STLContainerWriter( const Reflex::Type& objectType,
                                             MappingElement& mapping,
                                             ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_recordId(),
  m_localElement(),
  m_associative( ClassUtils::isTypeAssociativeContainer( objectType ) ),
  m_offset( 0 ),
  m_insertOperation( 0 ),
  m_arrayHandler(),
  m_keyWriter(),
  m_dataWriter(){
}

ora::STLContainerWriter::~STLContainerWriter(){
}

bool ora::STLContainerWriter::build( DataElement& offset,
                                     IRelationalData&,
                                     RelationalBuffer& operationBuffer ){
  m_localElement.clear();
  m_recordId.clear();
  // allocate for the index...
  m_recordId.push_back(0);
    
  RelationalStreamerFactory streamerFactory( m_schema );
  
  // first open the insert on the extra table...
  m_insertOperation = &operationBuffer.newBulkInsert( m_mappingElement.tableName() );
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( !columns.size() ){
    throwException( "Id columns not found in the mapping.",
                    "STLContainerWriter::build");    
  }
  for( size_t i=0; i<columns.size(); i++ ){
    m_insertOperation->addId( columns[ i ] );
  }

  m_offset = &offset;

  m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( m_objectType ) );
  
  Reflex::Type valueType;
  if ( m_associative ){
    
    Reflex::Type keyType = ClassUtils::containerKeyType(m_objectType);
    Reflex::Type keyResolvedType = ClassUtils::resolvedType(keyType);
    if ( ! keyType || !keyResolvedType ) {
      throwException( "Missing dictionary information for the key type of the container \"" +
                      m_objectType.Name(Reflex::SCOPED) + "\"",
                      "STLContainerWriter::build" );
    }
    std::string keyName = keyType.Name();
    // Retrieve the relevant mapping element
    MappingElement::iterator iMe = m_mappingElement.find( keyName );
    if ( iMe == m_mappingElement.end() ) {
      throwException( "Item for \"" + keyName + "\" not found in the mapping element",
                      "STLContainerWriter::build" );
    }

    m_keyWriter.reset( streamerFactory.newWriter( keyResolvedType, iMe->second ) );
    m_keyWriter->build( m_localElement, *m_insertOperation, operationBuffer );
    valueType = ClassUtils::containerDataType(m_objectType);
  } else {
    valueType = ClassUtils::containerValueType(m_objectType);
  }

  Reflex::Type valueResolvedType = ClassUtils::resolvedType(valueType);
  // Check the component type
  if ( ! valueType || !valueResolvedType ) {
    throwException( "Missing dictionary information for the content type of the container \"" +
                    m_objectType.Name(Reflex::SCOPED) + "\"",
                    "STLContainerWriter::build" );
  }
  
  std::string valueName = valueType.Name();
  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( valueName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + valueName + "\" not found in the mapping element",
                    "STLContainerWriter::build" );
  }

  m_dataWriter.reset( streamerFactory.newWriter( valueResolvedType, iMe->second ) );
  m_dataWriter->build( m_localElement, *m_insertOperation, operationBuffer );
  //operationBuffer.addToExecutionBuffer( *m_insertOperation );
  return true;
}

void ora::STLContainerWriter::setRecordId( const std::vector<int>& identity ){
  m_recordId.clear();
  for(size_t i=0;i<identity.size();i++) {
    m_recordId.push_back( identity[i] );
  }
  m_recordId.push_back( 0 );
}
      
void ora::STLContainerWriter::write( int oid,
                                     const void* inputData ){

  if(!m_offset){
    throwException("The streamer has not been built.",
                   "STLContainerWriter::write");
  }

  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( columns.size() != m_recordId.size()+1){
    throwException( "Object id elements provided are not matching with the mapped id columns.",
                    "STLContainerWriter::write");
  }

  const Reflex::Type& iteratorReturnType = m_arrayHandler->iteratorReturnType();
  // Retrieve the container type
  Reflex::Type keyType;
  if ( m_associative ) keyType = m_objectType.TemplateArgumentAt(0);
  Reflex::Member firstMember;
  Reflex::Member secondMember;
  if( keyType ){
    firstMember = iteratorReturnType.MemberByName( "first" );
    if ( ! firstMember ) {
      throwException( "Could not find the data member \"first\" for the class \"" +
                      iteratorReturnType.Name(Reflex::SCOPED) + "\"",
                      "STLContainerWriter::write" );
    }
    secondMember = iteratorReturnType.MemberByName( "second" );
    if ( ! secondMember ) {
      throwException( "Could not retrieve the data member \"second\" for the class \"" +
                      iteratorReturnType.Name(Reflex::SCOPED) + "\"",
                      "STLContainerWriter::write" );
    }
  }
  
  void* data = m_offset->address( inputData );
  
  // Use the iterator to loop over the elements of the container.
  size_t containerSize = m_arrayHandler->size( data  );
  size_t persistentSize = m_arrayHandler->persistentSize( data  );

  // TO BE CHECKED!!
  if ( containerSize == 0 || containerSize < persistentSize ) return;

  size_t startElementIndex = m_arrayHandler->startElementIndex( data );
  std::auto_ptr<IArrayIteratorHandler> iteratorHandler( m_arrayHandler->iterate( data ) );

  coral::IBulkOperation& bulkInsert = m_insertOperation->setUp( containerSize-startElementIndex+1 );

  for ( size_t iIndex = startElementIndex; iIndex < containerSize; ++iIndex ) {

    m_recordId[m_recordId.size()-1] = iIndex;
    coral::AttributeList& dataBuff = m_insertOperation->data();

    dataBuff[ columns[0] ].data<int>() = oid;
    for( size_t i = 1;i < columns.size(); i++ ){
      dataBuff[ columns[i] ].data<int>() = m_recordId[i-1];
    }

    void* objectReference = iteratorHandler->object();
    void* componentData = objectReference;

    if ( keyType ) { // treat the key object first
      void* keyData = static_cast< char* >( objectReference ) + firstMember.Offset();
      m_keyWriter->setRecordId( m_recordId );
      m_keyWriter->write( oid, keyData );

      componentData = static_cast< char* >( objectReference ) + secondMember.Offset();
    }
    m_dataWriter->setRecordId( m_recordId );

    m_dataWriter->write( oid, componentData );
    bulkInsert.processNextIteration();
   
    // Increment the iterator
    iteratorHandler->increment();
  }

  // execute the insert...
  m_arrayHandler->finalize( const_cast<void*>( data ) );

}

ora::STLContainerUpdater::STLContainerUpdater(const Reflex::Type& objectType,
                                              MappingElement& mapping,
                                              ContainerSchema& contSchema ):
  m_deleter( mapping ),
  m_writer( objectType, mapping, contSchema ){
}

ora::STLContainerUpdater::~STLContainerUpdater(){
}

bool ora::STLContainerUpdater::build( DataElement& offset,
                                      IRelationalData& relationalData,
                                      RelationalBuffer& operationBuffer){
  m_deleter.build( operationBuffer );
  m_writer.build( offset, relationalData, operationBuffer );
  return true;
}

void ora::STLContainerUpdater::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}

void ora::STLContainerUpdater::update( int oid,
                                       const void* data ){
  m_deleter.erase( oid );
  m_writer.write( oid, data );
}

ora::STLContainerReader::STLContainerReader(const Reflex::Type& objectType,
                                            MappingElement& mapping,
                                            ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_recordId(),
  m_localElement(),
  m_associative( ClassUtils::isTypeAssociativeContainer( objectType ) ),
  m_offset(0 ),
  m_query(),
  m_arrayHandler(),
  m_keyReader(),
  m_dataReader(){
}

ora::STLContainerReader::~STLContainerReader(){
}

bool ora::STLContainerReader::build( DataElement& offset, IRelationalData& ){
  m_localElement.clear();
  m_recordId.clear();
  // allocate for the index...
  m_recordId.push_back(0);

  RelationalStreamerFactory streamerFactory( m_schema );
  
  // first open the insert on the extra table...
  m_query.reset( new MultiRecordSelectOperation( m_mappingElement.tableName(), m_schema.storageSchema() ));
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  size_t stColIdx = m_mappingElement.startIndexForPKColumns();
  size_t cols = columns.size();
  if( cols==0 || cols < stColIdx+1 ){
    throwException( "Expected id column names have not been found in the mapping.",
                    "STLContainerReader::build");
  }

  m_query->addWhereId( columns[stColIdx] );
  for( size_t i=stColIdx+1; i<cols; i++ ){
    m_query->addId( columns[ i ] );
    m_query->addOrderId( columns[ i ] );
  }
  
  m_offset = &offset;

  m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( m_objectType ) );

  Reflex::Type valueType;
  if ( m_associative ){

    Reflex::Type keyType = ClassUtils::containerKeyType( m_objectType );
    Reflex::Type keyResolvedType = ClassUtils::resolvedType(keyType);

    if ( ! keyType ||!keyResolvedType ) {
      throwException( "Missing dictionary information for the key type of the container \"" +
                      m_objectType.Name(Reflex::SCOPED) + "\"",
                      "STLContainerReader::build" );
    }

    std::string keyName = keyType.Name();
    // Retrieve the relevant mapping element
    MappingElement::iterator iMe = m_mappingElement.find( keyName );
    if ( iMe == m_mappingElement.end() ) {
      throwException( "Item for \"" + keyName + "\" not found in the mapping element",
                      "STLContainerReader::build" );
    }

    m_keyReader.reset( streamerFactory.newReader( keyResolvedType, iMe->second ) );
    m_keyReader->build( m_localElement, *m_query );
    
    valueType = ClassUtils::containerDataType(m_objectType);
  } else {
    valueType = ClassUtils::containerValueType(m_objectType);
  }

  Reflex::Type valueResolvedType = ClassUtils::resolvedType(valueType);
  // Check the component type
  if ( ! valueType ||!valueResolvedType ) {
    throwException( "Missing dictionary information for the content type of the container \"" +
                    m_objectType.Name(Reflex::SCOPED) + "\"",
                    "STLContainerReader::build" );
  }
  
  std::string valueName = valueType.Name();
  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( valueName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + valueName + "\" not found in the mapping element",
                    "STLContainerReader::build" );
  }

  m_dataReader.reset( streamerFactory.newReader( valueResolvedType, iMe->second ) );
  m_dataReader->build( m_localElement, *m_query );
  return true;
}

void ora::STLContainerReader::select( int oid ){
  if(!m_query.get()){
    throwException("The streamer has not been built.",
                   "STLContainerReader::read");
  }
  coral::AttributeList& whereData = m_query->whereData();
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  size_t stColIdx = m_mappingElement.startIndexForPKColumns();
  whereData[ columns[ stColIdx ] ].data<int>() = oid;
  m_query->execute();
  if(m_keyReader.get()) m_keyReader->select( oid );
  m_dataReader->select( oid );
}

void ora::STLContainerReader::setRecordId( const std::vector<int>& identity ){
  m_recordId.clear();
  for(size_t i=0;i<identity.size();i++) {
    m_recordId.push_back( identity[i] );
  }
  // allocate the element for the index...
  m_recordId.push_back( 0 );
}

void ora::STLContainerReader::read( void* destinationData ) {

  if(!m_offset){
    throwException("The streamer has not been built.",
                   "STLContainerReader::read");
  }

  void* address = m_offset->address( destinationData );

  const Reflex::Type& iteratorReturnType = m_arrayHandler->iteratorReturnType();
  U_Primitives primitiveStub;
  
  Reflex::Type keyType;
  Reflex::Member firstMember;
  Reflex::Member secondMember;
  if ( m_associative ) {
    keyType = m_objectType.TemplateArgumentAt(0);
    firstMember = iteratorReturnType.MemberByName( "first" );
    if ( ! firstMember ) {
      throwException("Could not retrieve the data member \"first\" of the class \"" +
                     iteratorReturnType.Name(Reflex::SCOPED) + "\"",
                     "STLContainerReader::read" );
    }
    secondMember = iteratorReturnType.MemberByName( "second" );
    if ( ! secondMember ) {
      throwException( "Could not retrieve the data member \"second\" of the class \"" +
                      iteratorReturnType.Name(Reflex::SCOPED) + "\"",
                      "STLContainerReader::read" );
    }
  }
  
  bool isElementFundamental = iteratorReturnType.IsFundamental();
  
  m_arrayHandler->clear( address );

  size_t cursorSize = m_query->selectionSize(m_recordId, m_recordId.size()-1);
  unsigned int i=0;
  while ( i< cursorSize ){

    m_recordId[m_recordId.size()-1] = (int)i;
    m_query->selectRow( m_recordId );

    // Create a new element for the array
    void* objectData = 0;
    if(isElementFundamental){
      objectData = &primitiveStub;
    } else {
      objectData = iteratorReturnType.Construct().Address();
    }

    void* componentData = objectData;
    void* keyData = 0;

    if ( keyType ) { // treat the key object first
      keyData = static_cast< char* >( objectData ) + firstMember.Offset();
      m_keyReader->setRecordId( m_recordId );
      m_keyReader->read( keyData );
      
      componentData = static_cast< char* >( objectData ) + secondMember.Offset();
    }
    m_dataReader->setRecordId( m_recordId );
    m_dataReader->read( componentData );

    size_t prevSize = m_arrayHandler->size( address );
    m_arrayHandler->appendNewElement( address, objectData );
    bool inserted = m_arrayHandler->size( address )>prevSize;
    if ( ! ( iteratorReturnType.IsFundamental() ) ) {
      iteratorReturnType.Destruct( objectData );
    }
    if ( !inserted ) {
      throwException( "Could not insert a new element in the array type \"" +
                      m_objectType.Name(Reflex::SCOPED) + "\"",
                      "STLContainerReader::read" );
    }
    ++i;
  }

  m_arrayHandler->finalize( address );

}

ora::STLContainerStreamer::STLContainerStreamer( const Reflex::Type& objectType,
                                                 MappingElement& mapping,
                                                 ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::STLContainerStreamer::~STLContainerStreamer(){
}

ora::IRelationalWriter* ora::STLContainerStreamer::newWriter(){
  return new STLContainerWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::STLContainerStreamer::newUpdater(){
  return new STLContainerUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::STLContainerStreamer::newReader(){
  return new STLContainerReader( m_objectType, m_mapping, m_schema );
}
