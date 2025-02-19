#include "CondCore/ORA/interface/Exception.h"
#include "PrimitiveStreamer.h"
#include "DataElement.h"
#include "MappingElement.h"
#include "RelationalOperation.h"
#include "ClassUtils.h"
// externals
#include "CoralBase/Attribute.h"


ora::PrimitiveStreamerBase::PrimitiveStreamerBase( const Reflex::Type& objectType,
                                                   MappingElement& mapping ):
  m_objectType( objectType ),
  m_mapping(mapping),
  m_columnIndex(-1),
  m_dataElement( 0 ),
  m_relationalData( 0 ){  
}

ora::PrimitiveStreamerBase::~PrimitiveStreamerBase(){
}

bool ora::PrimitiveStreamerBase::buildDataElement(DataElement& dataElement,
                                                  IRelationalData& relationalData){
  if( m_mapping.columnNames().size()==0 ){
    throwException( "The mapping element does not contain columns.",
                    "PrimitiveStreamerBase::buildDataElement");
  }

  const std::type_info* attrType = &m_objectType.TypeInfo();
  if(m_objectType.IsEnum()) attrType = &typeid(int);
  if(ClassUtils::isTypeString( m_objectType )) attrType = &typeid(std::string);
  std::string columnName = m_mapping.columnNames()[0];
  m_columnIndex = relationalData.addData( columnName, *attrType );
  m_dataElement = &dataElement;
  m_relationalData = &relationalData;
  return true;
}

void ora::PrimitiveStreamerBase::bindDataForUpdate( const void* data ){
  if( ! m_dataElement ){
    throwException( "The streamer has not been built.",
                    "PrimitiveStreamerBase::bindDataForUpdate");
  }
  void* dataElementAddress = m_dataElement->address( data );
  coral::Attribute& relDataElement = m_relationalData->data()[ m_columnIndex ];
  relDataElement.setValueFromAddress( dataElementAddress );
  if(!relDataElement.isValidData()){
    throwException("Data provided for column \""+
                   relDataElement.specification().name()+
                   "\" is not valid for RDBMS storage.",
                   "PrimitiveStreamerBase::bindDataForUpdate");
  }  
}

void ora::PrimitiveStreamerBase::bindDataForRead( void* data ){
  if( ! m_dataElement ){
    throwException( "The streamer has not been built.",
                    "PrimitiveStreamerBase::bindDataForRead");
  }
  void* dataElementAddress = m_dataElement->address( data );
  coral::Attribute& relDataElement = m_relationalData->data()[ m_columnIndex ];
  relDataElement.copyValueToAddress( dataElementAddress );
}


ora::PrimitiveWriter::PrimitiveWriter( const Reflex::Type& objectType,
                                       MappingElement& mapping ):
  PrimitiveStreamerBase( objectType, mapping ){
}

ora::PrimitiveWriter::~PrimitiveWriter(){
}

bool ora::PrimitiveWriter::build(DataElement& dataElement,
                                 IRelationalData& relationalData,
                                 RelationalBuffer&){
  return buildDataElement( dataElement, relationalData );
}

void ora::PrimitiveWriter::setRecordId( const std::vector<int>& ){
}

void ora::PrimitiveWriter::write( int, const void* data ){
  bindDataForUpdate( data );  
}

ora::PrimitiveUpdater::PrimitiveUpdater( const Reflex::Type& objectType,
                                         MappingElement& mapping ):
  PrimitiveStreamerBase( objectType, mapping ){
}

ora::PrimitiveUpdater::~PrimitiveUpdater(){
}

bool ora::PrimitiveUpdater::build(DataElement& dataElement,
                                  IRelationalData& relationalData,
                                  RelationalBuffer&){
  return buildDataElement( dataElement, relationalData );  
}

void ora::PrimitiveUpdater::setRecordId( const std::vector<int>& ){
}

void ora::PrimitiveUpdater::update( int,
                                    const void* data ){
  bindDataForUpdate( data );  
}

ora::PrimitiveReader::PrimitiveReader( const Reflex::Type& objectType,
                                       MappingElement& mapping ):
  PrimitiveStreamerBase( objectType, mapping ){
}

ora::PrimitiveReader::~PrimitiveReader(){
}

bool ora::PrimitiveReader::build(DataElement& dataElement,
                                 IRelationalData& relationalData){
  return buildDataElement( dataElement, relationalData );  
}

void ora::PrimitiveReader::select( int ){
}

void ora::PrimitiveReader::setRecordId( const std::vector<int>& ){
}

void ora::PrimitiveReader::read( void* data ){
  bindDataForRead( data );
}

void ora::PrimitiveReader::clear(){
}


ora::PrimitiveStreamer::PrimitiveStreamer( const Reflex::Type& objectType,
                                           MappingElement& mapping ):
  m_objectType( objectType ),
  m_mapping( mapping ){
}

ora::PrimitiveStreamer::~PrimitiveStreamer(){
}

ora::IRelationalWriter* ora::PrimitiveStreamer::newWriter(){
  return new PrimitiveWriter( m_objectType, m_mapping );
}

ora::IRelationalUpdater* ora::PrimitiveStreamer::newUpdater(){
  return new PrimitiveUpdater( m_objectType, m_mapping );
}

ora::IRelationalReader* ora::PrimitiveStreamer::newReader(){
  return new PrimitiveReader( m_objectType, m_mapping );
}
