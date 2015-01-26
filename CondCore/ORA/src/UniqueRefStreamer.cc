#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/UniqueRef.h"
#include "UniqueRefStreamer.h"
#include "RelationalOperation.h"
#include "MappingElement.h"
#include "MappingRules.h"
#include "ContainerSchema.h"
#include "ClassUtils.h"
#include "RelationalStreamerFactory.h"
// externals
#include "CoralBase/Attribute.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

namespace ora {
  
  class DependentClassWriter {
    public:

      DependentClassWriter( ):
        m_dataElement(),
        m_writer(),
        m_depInsert( 0 ){
      }

      void build( const edm::TypeWithDict& objectType, MappingElement& mapping,
                  ContainerSchema& contSchema, RelationalBuffer& operationBuffer ){

        m_depInsert = &operationBuffer.newInsert( mapping.tableName());
        const std::vector<std::string> columns = mapping.columnNames();
        for( std::vector<std::string>::const_iterator iC = columns.begin();
             iC != columns.end(); ++iC){
          m_depInsert->addId( *iC );
        }

        MappingElement::iterator iMe = mapping.find( objectType.cppName() );
        // the first inner mapping is the relevant...
        if( iMe == mapping.end()){
          throwException("Could not find a mapping element for class \""+
                         objectType.cppName()+"\"",
                         "DependentClassWriter::write");
        }
        RelationalStreamerFactory streamerFactory( contSchema );
        m_writer.reset( streamerFactory.newWriter( objectType, iMe->second ) );
        m_writer->build( m_dataElement, *m_depInsert, operationBuffer );
      }

      ~DependentClassWriter(){
      }

      void write( int oId, int refId, const void* data ){
        if( !m_depInsert ){
          throwException( "DependentClassWriter has not been built.",
                          "DependentClassWriter::write");
        }
        
        coral::AttributeList& dataBuff = m_depInsert->data();
        coral::AttributeList::iterator iData = dataBuff.begin();
        iData->data<int>()= oId;
        ++iData;
        iData->data<int>()= refId;
        // transfering the refId
        std::vector<int> recordId(1,refId);
        m_writer->setRecordId( recordId);
        m_writer->write( oId, data );
      }

    private:
      DataElement m_dataElement;
      std::auto_ptr<IRelationalWriter> m_writer;
      InsertOperation* m_depInsert;
  };
  
  class DependentClassReader {
    public:
      DependentClassReader():
        m_dataElement(),
        m_type(),
        m_reader(),
        m_depQuery(){
      }

      void build( const edm::TypeWithDict& objectType, MappingElement& depMapping, ContainerSchema& contSchema ){

        m_type = objectType;
        m_depQuery.reset( new SelectOperation( depMapping.tableName(), contSchema.storageSchema()));
        m_depQuery->addWhereId(  depMapping.columnNames()[ 1 ] );
        MappingElement::iterator iMap = depMapping.find( m_type.cppName() );
        // the first inner mapping is the good one ...
        if( iMap == depMapping.end()){
          throwException("Could not find a mapping element for class \""+
                         m_type.cppName()+"\"",
                         "DependentClassReadBuffer::ReadBuffer");
        }
        MappingElement& mapping = iMap->second;
        RelationalStreamerFactory streamerFactory( contSchema );
        m_reader.reset( streamerFactory.newReader( m_type, mapping )) ;
        m_reader->build( m_dataElement , *m_depQuery );
      }

      ~DependentClassReader(){
      }

      void* read( int refId ){
        if( !m_depQuery.get() ){
          throwException( "DependentClassReader has not been built.",
                          "DependentClassReader::read");
        }
        coral::AttributeList& whereData = m_depQuery->whereData();
        coral::AttributeList::iterator iWData = whereData.begin();
        iWData->data<int>() = refId;
        m_depQuery->execute();
        m_reader->select( refId );
        void* destination = 0;
        if( m_depQuery->nextCursorRow() ){
          destination = ClassUtils::constructObject( m_type );
          m_reader->read( destination );
        }
        m_depQuery->clear();
        m_reader->clear();
        return destination;
      }

    private:
      DataElement m_dataElement;
      edm::TypeWithDict m_type;
      std::auto_ptr<IRelationalReader> m_reader;
      std::auto_ptr<SelectOperation> m_depQuery;
  };
    
  class RelationalRefLoader: public IPtrLoader {
    public:
      explicit RelationalRefLoader( int refId ):
        m_reader(),
        m_refId( refId ),
        m_valid( false ){
      }
      
      virtual ~RelationalRefLoader(){
      }
      
    public:
      void build( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema ){
        m_reader.build( objectType, mapping, contSchema );
        m_valid = true;
      }
      
      void* load() const override {
        if(!m_valid){
          throwException("Ref Loader has been invalidate.",
                         "RelationalRefLoader::load");
        }
        return m_reader.read( m_refId );
      }

      void invalidate() override{
        m_valid = false;
      }

      bool isValid() const override{
        return m_valid;
      }
      
    private:
      mutable DependentClassReader m_reader;
      int m_refId;
      bool m_valid;
  };    
}

ora::UniqueRefWriter::UniqueRefWriter( const edm::TypeWithDict& objectType,
                                       MappingElement& mapping,
                                       ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_dataElement( 0 ),
  m_relationalData( 0 ),
  m_operationBuffer( 0 ){
  m_columnIndexes[0]=-1;  
  m_columnIndexes[1]=-1;  
}

std::string ora::uniqueRefNullLabel(){
  static const std::string nullLabel("ora::UniqueRef::Null");
  return nullLabel;
}
      
ora::UniqueRefWriter::~UniqueRefWriter(){
}

bool ora::UniqueRefWriter::build(DataElement& dataElement,
                                 IRelationalData& relationalData,
                                 RelationalBuffer& buffer){
  m_dataElement = &dataElement;

  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( columns.size() < 2 ){
    throwException("Expected column names have not been found in the mapping.",
		   "UniqueRefWriter::build");    
  }
  // booking for ref metadata 
  m_columnIndexes[0] = relationalData.addData( columns[0],typeid(std::string) );
  // booking for ref id 
  m_columnIndexes[1] = relationalData.addData( columns[1],typeid(int) );
  m_relationalData = &relationalData;
  m_operationBuffer = &buffer;
  return true;
}

void ora::UniqueRefWriter::setRecordId( const std::vector<int>& ){
}

/// Writes a data element
void ora::UniqueRefWriter::write( int oid,
                                  const void* data ){

  if(!m_dataElement){
    throwException("The streamer has not been built.",
                   "UniqueRefWriter::write");
  }

  void* refAddress = m_dataElement->address( data );

  edm::ObjectWithDict refObj( m_objectType, const_cast<void*>(refAddress));

  bool isNull;
  edm::ObjectWithDict resObj = edm::ObjectWithDict(edm::TypeWithDict(typeid(bool)), &isNull);
  refObj.typeOf().functionMemberByName("operator!").invoke(refObj, &resObj);

  int refId = 0;
  std::string className = uniqueRefNullLabel();

  if(!isNull){
    // resolving the ref type
    const std::type_info *refTypeInfo = 0;
    edm::TypeWithDict typeInfoDict( typeid(std::type_info*) );
    edm::ObjectWithDict refTIObj( typeInfoDict, &refTypeInfo );
    m_objectType.functionMemberByName("typeInfo").invoke( refObj,  &refTIObj);
    edm::TypeWithDict refType = ClassUtils::lookupDictionary( *refTypeInfo );
    className = refType.cppName();

    // building the dependent buffer
    MappingElement& depMapping =  m_schema.mappingForDependentClass( refType, true );    

    // getting a new valid ref id
    refId = m_schema.containerSequences().getNextId( MappingRules::sequenceNameForDependentClass( m_schema.containerName(),className ));
    
    DependentClassWriter writer;
    writer.build( refType, depMapping, m_schema, m_operationBuffer->addVolatileBuffer() );
    void* refData = 0;
    edm::ObjectWithDict refDataObj(edm::TypeWithDict(typeid(void*)), &refData);
    m_objectType.functionMemberByName("operator*").invoke( refObj, &refDataObj );
    
    writer.write( oid, refId, refData );

  }
  // writing in the parent table
  coral::AttributeList& parentData = m_relationalData->data();
  parentData[m_columnIndexes[0]].data<std::string>()=className;
  parentData[m_columnIndexes[1]].data<int>()=refId;
}

ora::UniqueRefUpdater::UniqueRefUpdater( const edm::TypeWithDict& objectType,
                                         MappingElement& mapping,
                                         ContainerSchema& contSchema ):
  m_writer( objectType, mapping, contSchema ){
}
      
ora::UniqueRefUpdater::~UniqueRefUpdater(){
}

bool ora::UniqueRefUpdater::build(DataElement& dataElement,
                                  IRelationalData& relationalData,
                                  RelationalBuffer& operationBuffer){
  return m_writer.build( dataElement, relationalData, operationBuffer );
}

void ora::UniqueRefUpdater::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}

/// Writes a data element
void ora::UniqueRefUpdater::update( int oid,
                                    const void* data ){
  m_writer.write( oid, data );
}

ora::UniqueRefReader::UniqueRefReader( const edm::TypeWithDict& objectType,
                                       MappingElement& mapping,
                                       ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_dataElement( 0 ),
  m_relationalData( 0 ),
  m_loaders(){
  m_columnIndexes[0]=-1;  
  m_columnIndexes[1]=-1;  
}
      
ora::UniqueRefReader::~UniqueRefReader(){
  for(std::vector<boost::shared_ptr<RelationalRefLoader> >::const_iterator iL = m_loaders.begin();
      iL != m_loaders.end(); ++iL ){
    (*iL)->invalidate();
  }
}

bool ora::UniqueRefReader::build( DataElement& dataElement,
                                  IRelationalData& relationalData){
  m_dataElement = &dataElement;
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( columns.size() < 2 ){
    throwException("Expected column names have not been found in the mapping.",
		   "UniqueRefReader::build");    
  }
  // booking for ref metadata 
  m_columnIndexes[0] = relationalData.addData( columns[0],typeid(std::string) );
  // booking for ref id 
  m_columnIndexes[1] = relationalData.addData( columns[1],typeid(int) );
  m_relationalData = &relationalData;
  return true;
}

void ora::UniqueRefReader::select( int ){
}

void ora::UniqueRefReader::setRecordId( const std::vector<int>& ){
}

/// Read a data element
void ora::UniqueRefReader::read( void* data ){
  if(!m_dataElement){
    throwException("The streamer has not been built.",
                   "UniqueRefReader::read");
  }
  coral::AttributeList& row = m_relationalData->data();
  std::string className = row[m_columnIndexes[0]].data<std::string>();
  int refId = row[m_columnIndexes[1]].data<int>();

  edm::TypeWithDict refType = ClassUtils::lookupDictionary(className);
  
  // building the dependent buffer
  MappingElement& depMapping =  m_schema.mappingForDependentClass( refType );
  
  // resolving loader address
  edm::MemberWithDict loaderMember = m_objectType.dataMemberByName("m_loader");
  DataElement& loaderElement = m_dataElement->addChild( loaderMember.offset(), 0 );
  void* loaderAddress = loaderElement.address( data );
  boost::shared_ptr<IPtrLoader>* loaderPtr = static_cast<boost::shared_ptr<IPtrLoader>*>( loaderAddress );
  // creating new loader
  boost::shared_ptr<RelationalRefLoader> newLoader( new RelationalRefLoader( refId ) );
  newLoader->build( refType, depMapping, m_schema );
  m_loaders.push_back( newLoader );
  // installing the new loader
  boost::shared_ptr<IPtrLoader> tmp( newLoader );
  *loaderPtr = tmp;
}

void ora::UniqueRefReader::clear(){
}

ora::UniqueRefStreamer::UniqueRefStreamer( const edm::TypeWithDict& objectType,
                                           MappingElement& mapping,
                                           ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::UniqueRefStreamer::~UniqueRefStreamer(){
}

ora::IRelationalWriter* ora::UniqueRefStreamer::newWriter(){
  return new UniqueRefWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::UniqueRefStreamer::newUpdater(){
  return new UniqueRefUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::UniqueRefStreamer::newReader(){
  return new UniqueRefReader( m_objectType, m_mapping, m_schema );
}

