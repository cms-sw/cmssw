#include "CondCore/ORA/interface/Exception.h"
#include "STLContainerStreamer.h"
#include "ClassUtils.h"
#include "MappingElement.h"
#include "DataElement.h"
#include "ContainerSchema.h"
#include "RelationalBuffer.h"
#include "MultiRecordInsertOperation.h"
#include "MultiRecordSelectOperation.h"
#include "RelationalStreamerFactory.h"
#include "ArrayHandlerFactory.h"
#include "IArrayHandler.h"
#include "ArrayCommonImpl.h"
// externals
#include "CoralBase/Attribute.h"
#include "RelationalAccess/IBulkOperation.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

ora::STLContainerWriter::STLContainerWriter( const edm::TypeWithDict& objectType,
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
  if( !m_objectType ){
    throwException( "Missing dictionary information for the type of the container \"" +
                    m_objectType.cppName() + "\"",
                    "STLContainerWriter::build" );
  }
  m_localElement.clear();
  m_recordId.clear();
  // allocate for the index...
  m_recordId.push_back(0);

  RelationalStreamerFactory streamerFactory( m_schema );

  // first open the insert on the extra table...
  m_insertOperation = &operationBuffer.newMultiRecordInsert( m_mappingElement.tableName() );
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

  edm::TypeWithDict valueType;
  if ( m_associative ){

    edm::TypeWithDict keyType = ClassUtils::containerKeyType(m_objectType);
    edm::TypeWithDict keyResolvedType = ClassUtils::resolvedType(keyType);
    if ( ! keyType || !keyResolvedType ) {
      throwException( "Missing dictionary information for the key type of the container \"" +
                      m_objectType.cppName() + "\"",
                      "STLContainerWriter::build" );
    }
    std::string keyName("key_type");
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

  edm::TypeWithDict valueResolvedType = ClassUtils::resolvedType(valueType);
  // Check the component type
  if ( ! valueType || !valueResolvedType ) {
    throwException( "Missing dictionary information for the content type of the container \"" +
                    m_objectType.cppName() + "\"",
                    "STLContainerWriter::build" );
  }

  std::string valueName(m_associative ? "mapped_type" : "value_type");
  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( valueName );
  if ( iMe == m_mappingElement.end() ) {
    // Try again with the name of a possible typedef
    std::string valueName2 = valueType.unscopedName();
    iMe = m_mappingElement.find( valueName2 );
    if ( iMe == m_mappingElement.end() ) {
      throwException( "Item for \"" + valueName + "\" not found in the mapping element",
                      "STLContainerWriter::build" );
    }
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

  const edm::TypeWithDict& iteratorReturnType = m_arrayHandler->iteratorReturnType();
  // Retrieve the container type
  edm::TypeWithDict keyType;
  if ( m_associative ) keyType = m_objectType.templateArgumentAt(0);
  edm::MemberWithDict firstMember;
  edm::MemberWithDict secondMember;
  if( keyType ){
    firstMember = iteratorReturnType.dataMemberByName( "first" );
    if ( ! firstMember ) {
      throwException( "Could not find the data member \"first\" for the class \"" +
                      iteratorReturnType.cppName() + "\"",
                      "STLContainerWriter::write" );
    }
    secondMember = iteratorReturnType.dataMemberByName( "second" );
    if ( ! secondMember ) {
      throwException( "Could not retrieve the data member \"second\" for the class \"" +
                      iteratorReturnType.cppName() + "\"",
                      "STLContainerWriter::write" );
    }
  }

  void* data = m_offset->address( inputData );

  // Use the iterator to loop over the elements of the container.
  size_t containerSize = m_arrayHandler->size( data  );

  if ( containerSize == 0 ) return;

  size_t startElementIndex = m_arrayHandler->startElementIndex( data );
  std::auto_ptr<IArrayIteratorHandler> iteratorHandler( m_arrayHandler->iterate( data ) );

  InsertCache& bulkInsert = m_insertOperation->setUp( containerSize-startElementIndex+1 );

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
      void* keyData = static_cast< char* >( objectReference ) + firstMember.offset();
      m_keyWriter->setRecordId( m_recordId );
      m_keyWriter->write( oid, keyData );

      componentData = static_cast< char* >( objectReference ) + secondMember.offset();
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

ora::STLContainerUpdater::STLContainerUpdater(const edm::TypeWithDict& objectType,
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

ora::STLContainerReader::STLContainerReader(const edm::TypeWithDict& objectType,
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

  m_query->addWhereId( m_mappingElement.pkColumn() );
  std::vector<std::string> recIdCols = m_mappingElement.recordIdColumns();
  for( size_t i=0; i<recIdCols.size(); i++ ){
    m_query->addId( recIdCols[ i ] );
    m_query->addOrderId( recIdCols[ i ] );
  }

  m_offset = &offset;

  m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( m_objectType ) );

  edm::TypeWithDict valueType;
  if ( m_associative ){

    edm::TypeWithDict keyType = ClassUtils::containerKeyType( m_objectType );
    edm::TypeWithDict keyResolvedType = ClassUtils::resolvedType(keyType);

    if ( ! keyType ||!keyResolvedType ) {
      throwException( "Missing dictionary information for the key type of the container \"" +
                      m_objectType.cppName() + "\"",
                      "STLContainerReader::build" );
    }

    std::string keyName("key_type");
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

  edm::TypeWithDict valueResolvedType = ClassUtils::resolvedType(valueType);
  // Check the component type
  if ( ! valueType ||!valueResolvedType ) {
    throwException( "Missing dictionary information for the content type of the container \"" +
                    m_objectType.cppName() + "\"",
                    "STLContainerReader::build" );
  }

  std::string valueName(m_associative ? "mapped_type" : "value_type");
  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( valueName );
  if ( iMe == m_mappingElement.end() ) {
    // Try again with the name of a possible typedef
    std::string valueName2 = valueType.unscopedName();
    iMe = m_mappingElement.find( valueName2 );
    if ( iMe == m_mappingElement.end() ) {
      throwException( "Item for \"" + valueName + "\" not found in the mapping element",
                      "STLContainerReader::build" );
    }
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
  whereData[ m_mappingElement.pkColumn() ].data<int>() = oid;
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

  const edm::TypeWithDict& iteratorReturnType = m_arrayHandler->iteratorReturnType();
  U_Primitives primitiveStub;

  edm::TypeWithDict keyType;
  edm::MemberWithDict firstMember;
  edm::MemberWithDict secondMember;
  if ( m_associative ) {
    keyType = m_objectType.templateArgumentAt(0);
    firstMember = iteratorReturnType.dataMemberByName( "first" );
    if ( ! firstMember ) {
      throwException("Could not retrieve the data member \"first\" of the class \"" +
                     iteratorReturnType.cppName() + "\"",
                     "STLContainerReader::read" );
    }
    secondMember = iteratorReturnType.dataMemberByName( "second" );
    if ( ! secondMember ) {
      throwException( "Could not retrieve the data member \"second\" of the class \"" +
                      iteratorReturnType.cppName() + "\"",
                      "STLContainerReader::read" );
    }
  }

  bool isElementFundamental = iteratorReturnType.isFundamental();

  m_arrayHandler->clear( address );

  size_t cursorSize = m_query->selectionSize(m_recordId, m_recordId.size()-1);
  unsigned int i=0;
  // Create a new element for the array
  void* objectData = 0;
  if(isElementFundamental){
    objectData = &primitiveStub;
  } else {
    objectData = iteratorReturnType.construct().address();
  }

  while ( i< cursorSize ){

    m_recordId[m_recordId.size()-1] = (int)i;
    m_query->selectRow( m_recordId );

    void* componentData = objectData;
    void* keyData = 0;

    if ( keyType ) { // treat the key object first
      keyData = static_cast< char* >( objectData ) + firstMember.offset();
      m_keyReader->setRecordId( m_recordId );
      m_keyReader->read( keyData );

      componentData = static_cast< char* >( objectData ) + secondMember.offset();
    }
    m_dataReader->setRecordId( m_recordId );
    m_dataReader->read( componentData );

    size_t prevSize = m_arrayHandler->size( address );
    m_arrayHandler->appendNewElement( address, objectData );
    bool inserted = m_arrayHandler->size( address )>prevSize;
    if ( !inserted ) {
      throwException( "Could not insert a new element in the array type \"" +
                      m_objectType.cppName() + "\"",
                      "STLContainerReader::read" );
    }
    ++i;
  }
  if ( ! ( iteratorReturnType.isFundamental() ) ) {
    iteratorReturnType.destruct( objectData );
  }

  m_arrayHandler->finalize( address );

}

void ora::STLContainerReader::clear(){
  if(m_query.get()) m_query->clear();
  if(m_keyReader.get()) m_keyReader->clear();
  if(m_dataReader.get()) m_dataReader->clear();
}

ora::STLContainerStreamer::STLContainerStreamer( const edm::TypeWithDict& objectType,
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
