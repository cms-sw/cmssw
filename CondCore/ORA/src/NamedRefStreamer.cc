#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/NamedRef.h"
#include "NamedRefStreamer.h"
#include "RelationalOperation.h"
#include "MappingElement.h"
#include "ContainerSchema.h"
#include "DatabaseSession.h"
#include "ClassUtils.h"
#include "RelationalStreamerFactory.h"
// externals
#include "CoralBase/Attribute.h"
#include "Reflex/Member.h"

std::string ora::namedRefNullLabel(){
  static std::string nullLabel("ora::NamedRef::Null");
  return nullLabel;
}

ora::NamedReferenceStreamerBase::NamedReferenceStreamerBase( const Reflex::Type& objectType,
                                                             MappingElement& mapping,
                                                             ContainerSchema& schema):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_columnIndex( -1 ),
  m_schema( schema ),
  m_dataElement( 0 ),
  m_refNameDataElement( 0 ),
  m_ptrDataElement( 0 ),
  m_flagDataElement( 0 ),
  m_relationalData( 0 ){
}

ora::NamedReferenceStreamerBase::~NamedReferenceStreamerBase(){
}


bool ora::NamedReferenceStreamerBase::buildDataElement(DataElement& dataElement,
                                                       IRelationalData& relationalData){
  m_dataElement = &dataElement;
  m_objectType.UpdateMembers();
  Reflex::Member nameMember = m_objectType.DataMemberByName("m_name");
  if( !nameMember ){
    throwException("Data member \"m_name\" not found in class \""+m_objectType.Name()+"\".",
                   "NamedReferenceStreamerBase::buildDataElement");
  }
  m_refNameDataElement = &dataElement.addChild( nameMember.Offset(), 0 );
  Reflex::Member ptrMember = m_objectType.DataMemberByName("m_ptr");
  if( !ptrMember ){
    throwException("Data member \"m_ptr\" not found in class \""+m_objectType.Name()+"\".",
                   "NamedReferenceStreamerBase::buildDataElement");
  }
  m_ptrDataElement = &dataElement.addChild( ptrMember.Offset(), 0 );
  Reflex::Member flagMember = m_objectType.DataMemberByName("m_isPersistent");
  if( !flagMember ){
    throwException("Data member \"m_isPersistent\" not found in class \""+m_objectType.Name()+"\".",
                   "NamedReferenceStreamerBase::buildDataElement");
  }
  m_flagDataElement = &dataElement.addChild( flagMember.Offset(), 0 );
  // then book the column in the data attribute... 
  const std::vector<std::string>& columns =  m_mapping.columnNames();
  if( columns.size()==0 ){
      throwException("No columns found in the mapping element",
                     "NamedReferenceStreamerBase::buildDataElement");    
  }  
  m_columnIndex = relationalData.addData( columns[0],  typeid(std::string) );
  m_relationalData = &relationalData;
  return true;
}

void ora::NamedReferenceStreamerBase::bindDataForUpdate( const void* data ){
  if(!m_relationalData){
    throwException("The streamer has not been built.",
                   "NamedReferenceStreamerBase::bindDataForUpdate");
  }
  
  void* refNameAddress = m_refNameDataElement->address( data );
  coral::Attribute& refNameAttr = m_relationalData->data()[ m_columnIndex ];
  std::string name = *static_cast<std::string*>(refNameAddress);
  if( name.empty() ) name = namedRefNullLabel();
  refNameAttr.data<std::string>()= name;
}

void ora::NamedReferenceStreamerBase::bindDataForRead( void* data ){
  if(!m_relationalData){
    throwException("The streamer has not been built.",
                   "NamedReferenceStreamerBase::bindDataForRead");
  }
  void* refNameAddress = m_refNameDataElement->address( data );
  void* ptrAddress = m_ptrDataElement->address( data );
  void* flagAddress = m_flagDataElement->address( data );
  coral::Attribute& refNameAttr = m_relationalData->data()[ m_columnIndex ];
  std::string name = refNameAttr.data<std::string>();
  if( name == namedRefNullLabel() ){
    name = std::string("");
  }
  if(!name.empty()){
    Reflex::Type namedRefType = m_objectType.TemplateArgumentAt(0);
    boost::shared_ptr<void> ptr = m_schema.dbSession().fetchTypedObjectByName( name, namedRefType );
    *static_cast<boost::shared_ptr<void>*>(ptrAddress) = ptr;
    *static_cast<bool*>(flagAddress) = true;
  }
  *static_cast<std::string*>(refNameAddress) = name;
}

ora::NamedRefWriter::NamedRefWriter( const Reflex::Type& objectType,
                                     MappingElement& mapping,
                                     ContainerSchema& contSchema ):
  NamedReferenceStreamerBase( objectType, mapping, contSchema ){
}
      
ora::NamedRefWriter::~NamedRefWriter(){
}

bool ora::NamedRefWriter::build(DataElement& dataElement,
                                IRelationalData& relationalData,
                                RelationalBuffer& operationBuffer){
  return buildDataElement( dataElement, relationalData );
}

void ora::NamedRefWriter::setRecordId( const std::vector<int>& identity ){
}

/// Writes a data element
void ora::NamedRefWriter::write( int oid,
                                 const void* data ){
  bindDataForUpdate( data );  
}

ora::NamedRefUpdater::NamedRefUpdater( const Reflex::Type& objectType,
                                       MappingElement& mapping,
                                       ContainerSchema& contSchema ):
  NamedReferenceStreamerBase( objectType, mapping, contSchema ){
}
      
ora::NamedRefUpdater::~NamedRefUpdater(){
}

bool ora::NamedRefUpdater::build(DataElement& dataElement,
                                 IRelationalData& relationalData,
                                 RelationalBuffer& operationBuffer){
  return buildDataElement( dataElement, relationalData );  
}

void ora::NamedRefUpdater::setRecordId( const std::vector<int>& identity ){
}

/// Writes a data element
void ora::NamedRefUpdater::update( int oid,
                                   const void* data ){
  bindDataForUpdate( data );  
}

ora::NamedRefReader::NamedRefReader( const Reflex::Type& objectType,
                                     MappingElement& mapping,
                                     ContainerSchema& contSchema ):
  NamedReferenceStreamerBase( objectType, mapping, contSchema ){
}
      
ora::NamedRefReader::~NamedRefReader(){
}

bool ora::NamedRefReader::build( DataElement& dataElement,
                                 IRelationalData& relationalData ){
  return buildDataElement( dataElement, relationalData );  
}

void ora::NamedRefReader::select( int oid ){
}

void ora::NamedRefReader::setRecordId( const std::vector<int>& identity ){
}

/// Read a data element
void ora::NamedRefReader::read( void* data ){
  bindDataForRead( data );
}

void ora::NamedRefReader::clear(){
}


ora::NamedRefStreamer::NamedRefStreamer( const Reflex::Type& objectType,
                                         MappingElement& mapping,
                                         ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::NamedRefStreamer::~NamedRefStreamer(){
}

ora::IRelationalWriter* ora::NamedRefStreamer::newWriter(){
  return new NamedRefWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::NamedRefStreamer::newUpdater(){
  return new NamedRefUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::NamedRefStreamer::newReader(){
  return new NamedRefReader( m_objectType, m_mapping, m_schema );
}

