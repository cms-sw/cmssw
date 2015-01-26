#include "CondCore/ORA/interface/Exception.h"
#include "InlineCArrayStreamer.h"
#include "DataElement.h"
#include "MappingElement.h"
#include "RelationalOperation.h"
#include "MappingRules.h"
#include "ClassUtils.h"
// externals
#include "CoralBase/Attribute.h"


ora::InlineCArrayStreamerBase::InlineCArrayStreamerBase( const edm::TypeWithDict& objectType,
                                                         MappingElement& mapping,
                                                         ContainerSchema& contSchema):
  m_objectType( objectType ),
  m_arrayType(),
  m_streamerFactory( contSchema ),
  m_mapping( mapping ){
}

ora::InlineCArrayStreamerBase::~InlineCArrayStreamerBase(){
}


bool ora::InlineCArrayStreamerBase::buildDataElement(DataElement& dataElement,
                                                     IRelationalData& relationalData,
                                                     RelationalBuffer* operationBuffer){
  m_arrayType = ClassUtils::resolvedType( m_objectType.toType() );  
  if ( ! m_arrayType ) {
    throwException( "Missing dictionary information for the element of array \"" +
                    m_objectType.cppName() + "\"",
                    "InlineCArrayStreamerBase::buildDataElement" );
  }
  // Loop over the elements of the array.
  for ( size_t i=0;i<m_objectType.maximumIndex(0U);++i){

    // Form the element name
    std::string arrayElementLabel = MappingRules::variableNameForArrayIndex( m_mapping.variableName(),i);

    // Retrieve the relevant mapping element
    MappingElement::iterator iMe = m_mapping.find( arrayElementLabel );
    if ( iMe == m_mapping.end() ) {
      throwException( "Mapping for Array Element \"" + arrayElementLabel + "\" not found in the mapping element",
                      "InlineCArrayStreamerBase::buildDataElement" );
    }
    MappingElement& arrayElementMapping = iMe->second;
    DataElement& arrayElement = dataElement.addChild( i*m_arrayType.size(), 0 );

    processArrayElement( arrayElement, relationalData, arrayElementMapping, operationBuffer );
  }
  return true;
}

ora::InlineCArrayWriter::InlineCArrayWriter( const edm::TypeWithDict& objectType,
                                             MappingElement& mapping,
                                             ContainerSchema& contSchema ):
  InlineCArrayStreamerBase( objectType, mapping, contSchema ),
  m_writers(){
}

ora::InlineCArrayWriter::~InlineCArrayWriter(){
  for( std::vector< IRelationalWriter* >::iterator iW = m_writers.begin();
       iW != m_writers.end(); ++iW ){
    delete *iW;
  }
  m_writers.clear();
}

void ora::InlineCArrayWriter::processArrayElement( DataElement& arrayElementOffset,
                                                   IRelationalData& relationalData,
                                                   MappingElement& arrayElementMapping,
                                                   RelationalBuffer* operationBuffer ){
  IRelationalWriter* arrayElementWriter = m_streamerFactory.newWriter( m_arrayType, arrayElementMapping );
  m_writers.push_back( arrayElementWriter );
  arrayElementWriter->build( arrayElementOffset, relationalData, *operationBuffer );
}

bool ora::InlineCArrayWriter::build(DataElement& dataElement,
                                    IRelationalData& relationalData,
                                    RelationalBuffer& operationBuffer){
  return buildDataElement( dataElement, relationalData, &operationBuffer );
}

void ora::InlineCArrayWriter::setRecordId( const std::vector<int>& identity ){
  for( std::vector< IRelationalWriter* >::iterator iW = m_writers.begin();
       iW !=  m_writers.end(); ++iW ){
    (*iW)->setRecordId( identity );
  }  
}

void ora::InlineCArrayWriter::write( int oid, const void* data ){
  for( std::vector< IRelationalWriter* >::iterator iW = m_writers.begin();
       iW !=  m_writers.end(); ++iW ){
    (*iW)->write( oid, data );
  }
}

ora::InlineCArrayUpdater::InlineCArrayUpdater( const edm::TypeWithDict& objectType,
                                               MappingElement& mapping,
                                               ContainerSchema& contSchema  ):
  InlineCArrayStreamerBase( objectType, mapping, contSchema  ),
  m_updaters(){
}

ora::InlineCArrayUpdater::~InlineCArrayUpdater(){
  for( std::vector< IRelationalUpdater* >::iterator iU = m_updaters.begin();
       iU != m_updaters.end(); ++iU ){
    delete *iU;
  }
  m_updaters.clear();
}

void ora::InlineCArrayUpdater::processArrayElement( DataElement& arrayElementOffset,
                                                    IRelationalData& relationalData,
                                                    MappingElement& arrayElementMapping,
                                                    RelationalBuffer* operationBuffer ){
  IRelationalUpdater* arrayElementUpdater = m_streamerFactory.newUpdater( m_arrayType, arrayElementMapping );
  m_updaters.push_back( arrayElementUpdater );
  arrayElementUpdater->build( arrayElementOffset, relationalData, *operationBuffer );
}

bool ora::InlineCArrayUpdater::build(DataElement& dataElement,
                                     IRelationalData& relationalData,
                                     RelationalBuffer& operationBuffer){
  return buildDataElement( dataElement, relationalData, &operationBuffer );  
}

void ora::InlineCArrayUpdater::setRecordId( const std::vector<int>&  identity ){
  for( std::vector< IRelationalUpdater* >::iterator iU = m_updaters.begin();
       iU !=  m_updaters.end(); ++iU){
    (*iU)->setRecordId( identity );
  }  
}

void ora::InlineCArrayUpdater::update( int oid,
                                       const void* data ){
  for( std::vector< IRelationalUpdater* >::iterator iU = m_updaters.begin();
       iU !=  m_updaters.end(); ++iU ){
    (*iU)->update( oid, data );
  }
}

ora::InlineCArrayReader::InlineCArrayReader( const edm::TypeWithDict& objectType,
                                             MappingElement& mapping,
                                             ContainerSchema& contSchema ):
  InlineCArrayStreamerBase( objectType, mapping, contSchema ),
  m_readers(){
}

ora::InlineCArrayReader::~InlineCArrayReader(){
  for( std::vector< IRelationalReader* >::iterator iStr = m_readers.begin();
       iStr != m_readers.end(); ++iStr ){
    delete *iStr;
  }
  m_readers.clear();
}

void ora::InlineCArrayReader::processArrayElement( DataElement& arrayElementOffset,
                                                   IRelationalData& relationalData,
                                                   MappingElement& arrayElementMapping,
                                                   RelationalBuffer*){
  IRelationalReader* arrayElementReader = m_streamerFactory.newReader( m_arrayType, arrayElementMapping );
  m_readers.push_back( arrayElementReader );
  arrayElementReader->build( arrayElementOffset, relationalData );
}

bool ora::InlineCArrayReader::build(DataElement& dataElement,
                                    IRelationalData& relationalData){
  return buildDataElement( dataElement, relationalData, 0 );  
}

void ora::InlineCArrayReader::select( int oid){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->select( oid );
  }
}

void ora::InlineCArrayReader::setRecordId( const std::vector<int>&  identity ){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->setRecordId( identity );
  }  
}

void ora::InlineCArrayReader::read( void* data ){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->read( data );
  }
}

void ora::InlineCArrayReader::clear(){
  for( std::vector< IRelationalReader* >::iterator iDepReader = m_readers.begin();
       iDepReader !=  m_readers.end(); ++iDepReader ){
    (*iDepReader)->clear();
  }
}
    
ora::InlineCArrayStreamer::InlineCArrayStreamer( const edm::TypeWithDict& objectType,
                                                 MappingElement& mapping,
                                                 ContainerSchema& schema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( schema ){
}

ora::InlineCArrayStreamer::~InlineCArrayStreamer(){
}

ora::IRelationalWriter* ora::InlineCArrayStreamer::newWriter(){
  return new InlineCArrayWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::InlineCArrayStreamer::newUpdater(){
  return new InlineCArrayUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::InlineCArrayStreamer::newReader(){
  return new InlineCArrayReader( m_objectType, m_mapping, m_schema );
}
