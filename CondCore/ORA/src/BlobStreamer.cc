#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IBlobStreamingService.h"
#include "DatabaseSession.h"
#include "IDatabaseSchema.h"
#include "BlobStreamer.h"
#include "DataElement.h"
#include "MappingElement.h"
#include "RelationalOperation.h"
#include "RelationalBuffer.h"
#include "ContainerSchema.h"
#include "ClassUtils.h"
// externals
#include "CoralBase/Attribute.h"
#include "CoralBase/Blob.h"

ora::BlobWriterBase::BlobWriterBase( const Reflex::Type& objectType,
                                     MappingElement& mapping,
                                     ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_columnIndex(-1),
  m_schema( contSchema ),
  m_dataElement( 0 ),
  m_relationalData( 0 ),
  m_relationalBuffer( 0 ),
  m_blobWriter( 0 ),
  m_useCompression( true ){
}

ora::BlobWriterBase::~BlobWriterBase(){
}

bool ora::BlobWriterBase::buildDataElement(DataElement& dataElement,
                                           IRelationalData& relationalData,
                                           RelationalBuffer& operationBuffer){
  if( m_mapping.columnNames().size() == 0 ){
    throwException( "The mapping element does not contain columns.",
                    "BlobWriterBase::build");
  }

  m_dataElement = &dataElement;
  std::string columnName = m_mapping.columnNames()[0];
  m_columnIndex = relationalData.addBlobData( columnName );
  m_relationalData = &relationalData;
  m_relationalBuffer = &operationBuffer;
  m_blobWriter = m_schema.blobStreamingService();
  if(!m_blobWriter){
    throwException("Blob Streaming Service is not installed.",
                   "BlobWriterBase::::build");
  }
  if( m_schema.dbSession().schema().mainTable().schemaVersion()==poolSchemaVersion() ) m_useCompression = false;
  return true;
}

void ora::BlobWriterBase::bindData( const void* data ){
  if( ! m_dataElement ){
    throwException( "The streamer has not been built.",
                    "BlobWriterBase::bindData");
  }
  void* dataElementAddress = m_dataElement->address( data );
  coral::Attribute& relDataElement = m_relationalData->data()[ m_columnIndex ];
  boost::shared_ptr<coral::Blob> blobData = m_blobWriter->write( dataElementAddress, m_objectType, m_useCompression );
  m_relationalBuffer->storeBlob( blobData );
  relDataElement.bind<coral::Blob>( *blobData );
}

ora::BlobWriter::BlobWriter( const Reflex::Type& objectType,
                             MappingElement& mapping,
                             ContainerSchema& contSchema ):
  BlobWriterBase( objectType, mapping, contSchema ){
}

ora::BlobWriter::~BlobWriter(){
}

bool ora::BlobWriter::build(DataElement& dataElement,
                            IRelationalData& relationalData,
                            RelationalBuffer& relationalBuffer ){
  return buildDataElement( dataElement, relationalData, relationalBuffer );
}

void ora::BlobWriter::setRecordId( const std::vector<int>& ){
}

void ora::BlobWriter::write( int,
                             const void* data ){
  bindData( data );  
}

ora::BlobUpdater::BlobUpdater( const Reflex::Type& objectType,
                               MappingElement& mapping,
                               ContainerSchema& contSchema ):
  BlobWriterBase( objectType, mapping, contSchema ){
}

ora::BlobUpdater::~BlobUpdater(){
}

bool ora::BlobUpdater::build(DataElement& dataElement,
                             IRelationalData& relationalData,
                             RelationalBuffer& relationalBuffer){
  return buildDataElement( dataElement, relationalData, relationalBuffer );  
}

void ora::BlobUpdater::setRecordId( const std::vector<int>& ){
}

void ora::BlobUpdater::update( int,
                               const void* data ){
  bindData( data );  
}


ora::BlobReader::BlobReader( const Reflex::Type& objectType,
                             MappingElement& mapping,
                             ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_columnIndex( -1 ),
  m_schema( contSchema ),
  m_dataElement( 0 ),
  m_relationalData( 0 ),
  m_blobReader( 0 ){
}

ora::BlobReader::~BlobReader(){
}

bool ora::BlobReader::build(DataElement& dataElement,
                            IRelationalData& relationalData){
  
  if( m_mapping.columnNames().size() == 0 ){
    throwException( "The mapping element does not contain columns.",
                    "BlobReader::build");
  }

  m_dataElement = &dataElement;
  std::string columnName = m_mapping.columnNames()[0];
  m_columnIndex = relationalData.addBlobData( columnName );
  m_relationalData = &relationalData;
  m_blobReader = m_schema.blobStreamingService();
  if(!m_blobReader){
    throwException("Blob Streaming Service is not installed.",
                   "BlobReader::build");
  }
  return true;
}

void ora::BlobReader::select( int ){
}

void ora::BlobReader::setRecordId( const std::vector<int>& ){
}

void ora::BlobReader::read( void* data ){
  if( ! m_dataElement ){
    throwException( "The streamer has not been built.",
                    "BlobReader::read");
  }
  void* dataElementAddress = m_dataElement->address( data );
  coral::Attribute& relDataElement = m_relationalData->data()[ m_columnIndex ];
  m_blobReader->read(relDataElement.data<coral::Blob>(), dataElementAddress, m_objectType );
}

void ora::BlobReader::clear(){
}

ora::BlobStreamer::BlobStreamer( const Reflex::Type& objectType,
                                 MappingElement& mapping,
                                 ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::BlobStreamer::~BlobStreamer(){
}

ora::IRelationalWriter* ora::BlobStreamer::newWriter(){
  return new BlobWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::BlobStreamer::newUpdater(){
  return new BlobUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::BlobStreamer::newReader(){
  return new BlobReader( m_objectType, m_mapping, m_schema );
}
