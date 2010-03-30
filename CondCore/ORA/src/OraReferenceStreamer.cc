#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Reference.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
#include "OraReferenceStreamer.h"
#include "DataElement.h"
#include "MappingElement.h"
#include "ContainerSchema.h"
#include "RelationalOperation.h"
#include "ClassUtils.h"
// externals
#include "CoralBase/Attribute.h"
#include "Reflex/Base.h"
#include "Reflex/Member.h"

namespace ora {
  bool findBaseType( Reflex::Type& type, Reflex::Type& baseType, Reflex::OffsetFunction& func ){
    bool found = false;
    for ( unsigned int i=0;i<type.BaseSize() && !found; i++){
      Reflex::Base base = type.BaseAt(i);
      Reflex::Type bt = ClassUtils::resolvedType( base.ToType() );
      if( bt == baseType ){
        func = base.OffsetFP();
        found = true;
      } else {
        found = findBaseType( bt, baseType, func );
      }
    }
    return found;
  }
}

ora::OraReferenceStreamerBase::OraReferenceStreamerBase( const Reflex::Type& objectType,
                                                         MappingElement& mapping,
                                                         ContainerSchema& schema):
  m_objectType( objectType ),
  m_columns( mapping.columnNames() ),
  m_schema( schema ),
  m_dataElement( 0 ),
  m_dataElemOId0( 0 ),
  m_dataElemOId1( 0 ),
  m_relationalData( 0 ){
}

ora::OraReferenceStreamerBase::~OraReferenceStreamerBase(){
}


bool ora::OraReferenceStreamerBase::buildDataElement(DataElement& dataElement,
                                                     IRelationalData& relationalData){
  m_dataElement = &dataElement;
  // first resolve the oid0 and oid2 data elements...
  Reflex::Type refType = Reflex::Type::ByTypeInfo( typeid(Reference) );
  Reflex::Type oidType = Reflex::Type::ByTypeInfo( typeid(OId) );
  Reflex::OffsetFunction baseOffsetFunc = 0;
  if( m_objectType != refType ){
    bool foundRef = findBaseType( m_objectType, refType, baseOffsetFunc );
    if(!foundRef){
      throwException("Type \""+m_objectType.Name(Reflex::SCOPED)+"\" is not an Ora Reference.",
                     "OraReferenceStreamerBase::buildDataElement");
    } 
  }
  Reflex::Member contIdMember = refType.DataMemberByName("m_containerId");
  Reflex::Member itemIdMember = refType.DataMemberByName("m_itemId");
  if( !contIdMember || !itemIdMember ){
    throwException("Data members for class OId not found.",
                   "OraReferenceStreamerBase::buildDataElement");
  }
  m_dataElemOId0 = &dataElement.addChild( contIdMember.Offset(), baseOffsetFunc );
  m_dataElemOId1 = &dataElement.addChild( itemIdMember.Offset(), baseOffsetFunc);
  // then book the columns in the data attribute...  
  const std::type_info& attrType = typeid(int);
  for( std::vector<std::string>::const_iterator iCol = m_columns.begin();
       iCol != m_columns.end(); ++iCol ){
    relationalData.addData( *iCol, attrType );
  }
  m_relationalData = &relationalData;
  return true;
}


void ora::OraReferenceStreamerBase::bindDataForUpdate( const void* data ){
  if(!m_relationalData){
    throwException("The streamer has not been built.",
                   "OraReferenceStreamerBase::bindDataForUpdate");
  }
  if( m_columns.size()<2 ){
    throwException("Expected column names have not been found in the mapping.",
                   "OraReferenceStreamerBase::bindDataForUpdate");
  }
  
  void* oid0Address = m_dataElemOId0->address( data );
  coral::Attribute& oid0Attr = m_relationalData->data()[ m_columns[0] ];
  oid0Attr.data<int>()= *static_cast<int*>(oid0Address);
  void* oid1Address = m_dataElemOId1->address( data );
  coral::Attribute& oid1Attr = m_relationalData->data()[ m_columns[1] ];
  oid1Attr.data<int>()= *static_cast<int*>(oid1Address) ;
  IReferenceHandler* refHandler = m_schema.referenceHandler();
  void* refPtr = m_dataElement->address( data );
  if(refHandler) refHandler->onSave( *static_cast<Reference*>( refPtr ) );
}

void ora::OraReferenceStreamerBase::bindDataForRead( void* data ){
  if(!m_relationalData){
    throwException("The streamer has not been built.",
                   "OraReferenceStreamerBase::bindDataForRead");
  }
  if( m_columns.size()<2 ){
    throwException("Expected column names have not been found in the mapping.",
                   "OraReferenceStreamerBase::bindDataForRead");
  }
  void* oid0Address = m_dataElemOId0->address( data );
  coral::Attribute& oid0Attr = m_relationalData->data()[ m_columns[0] ];
  *static_cast<int*>(oid0Address) = oid0Attr.data<int>();
  void* oid1Address = m_dataElemOId1->address( data );
  coral::Attribute& oid1Attr = m_relationalData->data()[ m_columns[1] ];
  *static_cast<int*>( oid1Address ) = oid1Attr.data<int>();
  IReferenceHandler* refHandler = m_schema.referenceHandler();
  void* refPtr = m_dataElement->address( data );
  if(refHandler) refHandler->onLoad( *static_cast<Reference*>( refPtr ) );
}

ora::OraReferenceWriter::OraReferenceWriter( const Reflex::Type& objectType,
                                             MappingElement& mapping,
                                             ContainerSchema& schema  ):
  OraReferenceStreamerBase( objectType, mapping, schema ){
}

ora::OraReferenceWriter::~OraReferenceWriter(){
}

bool ora::OraReferenceWriter::build(DataElement& dataElement,
                                    IRelationalData& relationalData,
                                    RelationalBuffer&){
  return buildDataElement( dataElement, relationalData );
}

void ora::OraReferenceWriter::setRecordId( const std::vector<int>& ){
}

void ora::OraReferenceWriter::write( int,
                                     const void* data ){
  bindDataForUpdate( data );  
}

ora::OraReferenceUpdater::OraReferenceUpdater( const Reflex::Type& objectType,
                                               MappingElement& mapping,
                                               ContainerSchema& schema):
  OraReferenceStreamerBase( objectType, mapping, schema ){
}

ora::OraReferenceUpdater::~OraReferenceUpdater(){
}

bool ora::OraReferenceUpdater::build(DataElement& dataElement,
                                     IRelationalData& relationalData,
                                     RelationalBuffer&){
  return buildDataElement( dataElement, relationalData );  
}

void ora::OraReferenceUpdater::setRecordId( const std::vector<int>& ){
}

void ora::OraReferenceUpdater::update( int,
                                       const void* data ){
  bindDataForUpdate( data );  
}

ora::OraReferenceReader::OraReferenceReader( const Reflex::Type& objectType,
                                             MappingElement& mapping,
                                             ContainerSchema& schema ):
  OraReferenceStreamerBase( objectType, mapping, schema ){
}

ora::OraReferenceReader::~OraReferenceReader(){
}

bool ora::OraReferenceReader::build(DataElement& dataElement,
                                    IRelationalData& relationalData){
  return buildDataElement( dataElement, relationalData );  
}

void ora::OraReferenceReader::select( int ){
}

void ora::OraReferenceReader::setRecordId( const std::vector<int>& ){
}

void ora::OraReferenceReader::read( void* data ){
  bindDataForRead( data );
}

    
ora::OraReferenceStreamer::OraReferenceStreamer( const Reflex::Type& objectType,
                                                 MappingElement& mapping,
                                                 ContainerSchema& schema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( schema ){
}

ora::OraReferenceStreamer::~OraReferenceStreamer(){
}

ora::IRelationalWriter* ora::OraReferenceStreamer::newWriter(){
  return new OraReferenceWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::OraReferenceStreamer::newUpdater(){
  return new OraReferenceUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::OraReferenceStreamer::newReader(){
  return new OraReferenceReader( m_objectType, m_mapping, m_schema );
}
