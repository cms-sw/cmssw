#include "CondCore/ORA/interface/Exception.h"
#include "PVectorStreamer.h"
#include "ArrayCommonImpl.h"
#include "IArrayHandler.h"

ora::PVectorWriter::PVectorWriter( const Reflex::Type& objectType,
                                   MappingElement& mapping,
                                   ContainerSchema& contSchema ):
  m_writer( objectType, mapping, contSchema ){
}

ora::PVectorWriter::~PVectorWriter(){
}

bool ora::PVectorWriter::build( DataElement& offset,
                                IRelationalData& data,
                                RelationalBuffer& operationBuffer ){
  return m_writer.build( offset, data, operationBuffer );
}

void ora::PVectorWriter::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}
      
void ora::PVectorWriter::write( int oid,
                                const void* inputData )
{
  m_writer.write( oid, inputData );
}

ora::PVectorUpdater::PVectorUpdater(const Reflex::Type& objectType,
                                    MappingElement& mapping,
                                    ContainerSchema& contSchema ):
  m_buffer(0),
  m_writer( objectType, mapping, contSchema ){
}

ora::PVectorUpdater::~PVectorUpdater(){
}

bool ora::PVectorUpdater::build( DataElement& offset,
                                 IRelationalData& relationalData,
                                 RelationalBuffer& operationBuffer){
  m_buffer = &operationBuffer;
  return m_writer.build( offset, relationalData, operationBuffer );
}

void ora::PVectorUpdater::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}

void ora::PVectorUpdater::update( int oid,
                                  const void* data ){
  if(!m_writer.dataElement()){
    throwException("The streamer has not been built.",
                   "PVectorUpdater::update");
  }
  
  void* arrayData = m_writer.dataElement()->address( data );
  IArrayHandler& arrayHandler = *m_writer.arrayHandler();
  
  size_t arraySize = arrayHandler.size(arrayData);
  size_t persistentSize = arrayHandler.persistentSize(arrayData);
  if(persistentSize>arraySize){
    deleteArrayElements( m_writer.mapping(), oid, arraySize, *m_buffer );
  }
  else if(persistentSize<arraySize) {
    m_writer.write( oid, data );
  }
}

ora::PVectorReader::PVectorReader(const Reflex::Type& objectType,
                                  MappingElement& mapping,
                                  ContainerSchema& contSchema ):
  m_reader( objectType, mapping, contSchema ){
}

ora::PVectorReader::~PVectorReader(){
}

bool ora::PVectorReader::build( DataElement& offset,
                                IRelationalData& relationalData ){
  return m_reader.build( offset, relationalData );
}

void ora::PVectorReader::select( int oid ){
  m_reader.select( oid );
}

void ora::PVectorReader::setRecordId( const std::vector<int>& identity ){
  m_reader.setRecordId( identity );
}


void ora::PVectorReader::read( void* destinationData ) {
  m_reader.read( destinationData );
}

ora::PVectorStreamer::PVectorStreamer( const Reflex::Type& objectType,
                                       MappingElement& mapping,
                                       ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::PVectorStreamer::~PVectorStreamer(){
}

ora::IRelationalWriter* ora::PVectorStreamer::newWriter(){
  return new PVectorWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::PVectorStreamer::newUpdater(){
  return new PVectorUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::PVectorStreamer::newReader(){
  return new PVectorReader( m_objectType, m_mapping, m_schema );
}
