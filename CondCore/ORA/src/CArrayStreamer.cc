#include "CondCore/ORA/interface/Exception.h"
#include "CArrayStreamer.h"
#include "ClassUtils.h"
#include "MappingElement.h"
#include "ContainerSchema.h"
#include "RelationalBuffer.h"
#include "MultiRecordInsertOperation.h"
#include "MultiRecordSelectOperation.h"
#include "RelationalStreamerFactory.h"
#include "ArrayHandlerFactory.h"
#include "IArrayHandler.h"
// externals
#include "CoralBase/Attribute.h"
#include "RelationalAccess/IBulkOperation.h"
#include "Reflex/Object.h"

ora::CArrayWriter::CArrayWriter( const Reflex::Type& objectType,
                                 MappingElement& mapping,
                                 ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_recordId(),
  m_localElement( ),
  m_offset( 0 ),
  m_insertOperation( 0 ),
  m_arrayHandler(),
  m_dataWriter(){
}

ora::CArrayWriter::~CArrayWriter(){
}

bool ora::CArrayWriter::build( DataElement& offset,
                               IRelationalData&,
                               RelationalBuffer& operationBuffer ){

  m_localElement.clear();
  m_recordId.clear();
  // allocate for the index...
  m_recordId.push_back(0);

  // Check the array type
  Reflex::Type arrayType = m_objectType.ToType();
  Reflex::Type arrayResolvedType = ClassUtils::resolvedType(arrayType);
  // Check the component type
  if ( ! arrayType || !arrayResolvedType ) {
    throwException( "Missing dictionary information for the element type of the array \"" +
                    m_objectType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                    "CArrayWriter::build" );
  }
  
  RelationalStreamerFactory streamerFactory( m_schema );
  
  // first open the insert on the extra table...
  m_insertOperation = &operationBuffer.newMultiRecordInsert( m_mappingElement.tableName() );
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( !columns.size() ){
    throwException( "Id columns not found in the mapping.",
                    "CArrayWriter::build");    
  }
  for( size_t i=0; i<columns.size(); i++ ){
    m_insertOperation->addId( columns[ i ] );
  }

  m_offset = &offset;

  m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( m_objectType ) );
  
  std::string arrayTypeName = arrayType.Name();
  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( arrayTypeName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + arrayTypeName + "\" not found in the mapping element",
                    "CArrayWriter::build" );
  }
  
  m_dataWriter.reset( streamerFactory.newWriter( arrayResolvedType, iMe->second ));
  m_dataWriter->build( m_localElement, *m_insertOperation, operationBuffer );
  return true;
}

void ora::CArrayWriter::setRecordId( const std::vector<int>& identity ){
  m_recordId.clear();
  for(size_t i=0;i<identity.size();i++) {
    m_recordId.push_back( identity[i] );
  }
  m_recordId.push_back( 0 );
}
      
void ora::CArrayWriter::write( int oid,
                               const void* inputData ){

  if(!m_offset){
    throwException("The streamer has not been built.",
                   "CArrayWriter::write");
  }
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( columns.size() != m_recordId.size()+1){
    throwException( "Record id elements provided are not matching with the mapped id columns.",
                    "CArrayWriter::write");
  }
  
  void* data = m_offset->address( inputData );
  
  // Use the iterator to loop over the elements of the container.
  size_t containerSize = m_arrayHandler->size( data  );
  
  if ( containerSize == 0  ) return;

  size_t startElementIndex = m_arrayHandler->startElementIndex( data );

  std::auto_ptr<IArrayIteratorHandler> iteratorHandler( m_arrayHandler->iterate( data ) );

  InsertCache& bulkOperation = m_insertOperation->setUp( containerSize-startElementIndex+1 );

  for ( size_t iIndex = startElementIndex; iIndex < containerSize; ++iIndex ) {

    m_recordId[m_recordId.size()-1] = iIndex;
    coral::AttributeList& dataBuff = m_insertOperation->data();

    dataBuff[ columns[0] ].data<int>() = oid;
    for( size_t i = 1;i < columns.size(); i++ ){
      dataBuff[ columns[i] ].data<int>() = m_recordId[i-1];
    }

    void* objectReference = iteratorHandler->object();

    m_dataWriter->setRecordId( m_recordId );
    m_dataWriter->write( oid, objectReference );
    
    bulkOperation.processNextIteration();
    // Increment the iterator
    iteratorHandler->increment();
  }

  m_arrayHandler->finalize( const_cast<void*>( data ) );
  
}

ora::CArrayUpdater::CArrayUpdater(const Reflex::Type& objectType,
                                  MappingElement& mapping,
                                  ContainerSchema& contSchema ):
  m_deleter( mapping ),
  m_writer( objectType, mapping, contSchema ){
}

ora::CArrayUpdater::~CArrayUpdater(){
}

bool ora::CArrayUpdater::build( DataElement& offset,
                                IRelationalData& relationalData,
                                RelationalBuffer& operationBuffer){
  m_deleter.build( operationBuffer );
  m_writer.build( offset, relationalData, operationBuffer );
  return true;
}

void ora::CArrayUpdater::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}

void ora::CArrayUpdater::update( int oid,
                                 const void* data ){
  m_deleter.erase( oid );
  m_writer.write( oid, data );
}

ora::CArrayReader::CArrayReader(const Reflex::Type& objectType,
                                MappingElement& mapping,
                                ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_recordId(),
  m_localElement( ),
  m_offset(0 ),
  m_query(),
  m_arrayHandler(),
  m_dataReader(){
}

ora::CArrayReader::~CArrayReader(){
}

bool ora::CArrayReader::build( DataElement& offset,
                               IRelationalData& relationalData ){
  
  m_localElement.clear();

  m_recordId.clear();
  // allocate for the index...
  m_recordId.push_back(0);

  // Check the array type
  Reflex::Type arrayType = m_objectType.ToType();
  Reflex::Type arrayResolvedType = ClassUtils::resolvedType(arrayType);
  // Check the component type
  if ( ! arrayType || !arrayResolvedType ) {
    throwException( "Missing dictionary information for the element type of the array \"" +
                    m_objectType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                    "CArrayReader::build" );
  }

  RelationalStreamerFactory streamerFactory( m_schema );
  
  // first open the insert on the extra table...
  m_query.reset(new MultiRecordSelectOperation( m_mappingElement.tableName(), m_schema.storageSchema() ));
  m_query->addWhereId( m_mappingElement.pkColumn() );
  std::vector<std::string> recIdCols = m_mappingElement.recordIdColumns();
  for( size_t i=0; i<recIdCols.size(); i++ ){
    m_query->addId( recIdCols[ i ] );
    m_query->addOrderId( recIdCols[ i ] );
  }
  
  m_offset = &offset;

  m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( m_objectType ) );

  std::string arrayTypeName = arrayType.Name();

  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( arrayTypeName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + arrayTypeName + "\" not found in the mapping element",
                    "CArrayReader::build" );
  }
  
  m_dataReader.reset( streamerFactory.newReader( arrayResolvedType, iMe->second ) );
  m_dataReader->build( m_localElement, *m_query );
  return true;
}

void ora::CArrayReader::select( int oid ){
  if(!m_query.get()){
    throwException("The streamer has not been built.",
                   "CArrayReader::select");
  }
  coral::AttributeList& whereData = m_query->whereData();
  whereData[ m_mappingElement.pkColumn() ].data<int>() = oid;
  m_query->execute();
  m_dataReader->select( oid );
}

void ora::CArrayReader::setRecordId( const std::vector<int>& identity ){
  m_recordId.clear();
  for(size_t i=0;i<identity.size();i++) {
    m_recordId.push_back( identity[i] );
  }
  // allocate the element for the index...
  m_recordId.push_back( 0 );
}

void ora::CArrayReader::read( void* destinationData ) {
  if(!m_offset){
    throwException("The streamer has not been built.",
                   "CArrayReader::read");
  }
  void* address = m_offset->address( destinationData );

  Reflex::Type iteratorDereferenceReturnType = m_arrayHandler->iteratorReturnType();
  
  bool isElementFundamental = iteratorDereferenceReturnType.IsFundamental();

  std::string positionColumn = m_mappingElement.posColumn();

  size_t arraySize = m_objectType.ArrayLength();
  
  m_arrayHandler->clear( address );

  size_t cursorSize = m_query->selectionSize(m_recordId, m_recordId.size()-1);
  unsigned int i=0;
  while ( i< cursorSize ){

    m_recordId[m_recordId.size()-1] = (int)i;
    m_query->selectRow( m_recordId );
    coral::AttributeList& row = m_query->data();

    int arrayIndex = row[positionColumn].data< int >();

    // Create a new element for the array
    void* objectData = 0;
    
    if(arrayIndex >= (int)arraySize){
      throwException("Found more element then array size.",
                     "CArrayReader::read");
                     
    }
    
    // the memory has been allocated already!
    objectData = static_cast<char*>(address)+arrayIndex*iteratorDereferenceReturnType.SizeOf();

    if(!isElementFundamental){
      // in this case the initialization is required: use default constructor...
      iteratorDereferenceReturnType.Construct(Reflex::Type(0,0),std::vector< void* >(),objectData);
    }

    m_dataReader->setRecordId( m_recordId );
    m_dataReader->read( objectData );
    
    ++i;
  }

  m_arrayHandler->finalize( address );

}

void ora::CArrayReader::clear(){
  if(m_dataReader.get()) m_dataReader->clear();
}

ora::CArrayStreamer::CArrayStreamer( const Reflex::Type& objectType,
                                   MappingElement& mapping,
                                   ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::CArrayStreamer::~CArrayStreamer(){
}

ora::IRelationalWriter* ora::CArrayStreamer::newWriter(){
  return new CArrayWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::CArrayStreamer::newUpdater(){
  return new CArrayUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::CArrayStreamer::newReader(){
  return new CArrayReader( m_objectType, m_mapping, m_schema );
}
