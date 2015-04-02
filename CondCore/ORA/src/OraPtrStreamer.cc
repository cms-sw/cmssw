#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Ptr.h"
#include "OraPtrStreamer.h"
#include "RelationalOperation.h"
#include "MappingElement.h"
#include "ContainerSchema.h"
#include "ClassUtils.h"
#include "RelationalStreamerFactory.h"
// externals
#include "CoralBase/Attribute.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"

namespace ora {
  class OraPtrReadBuffer {
    public:
      OraPtrReadBuffer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema ):
        m_objectType( objectType ),
        m_mapping( mapping ),
        m_schema( contSchema ),
        m_localElement(),
        m_query( mapping.tableName(), contSchema.storageSchema() ),
        m_dataElement( 0 ),
        m_reader(){
      }

      ~OraPtrReadBuffer(){
      }
      
      bool build( DataElement& dataElement ){
        m_dataElement = &dataElement;
        m_localElement.clear();

        const std::vector<std::string>& columns = m_mapping.columnNames();
        size_t cols = columns.size();
        for( size_t i=0; i<cols; i++ ){
          m_query.addWhereId( columns[ i ] );
        }
        
        // Check the  type
        edm::TypeWithDict ptrType = m_objectType.templateArgumentAt(0);
        edm::TypeWithDict ptrResolvedType = ClassUtils::resolvedType(ptrType);
        // Check the component type
        if ( ! ptrType || !ptrResolvedType ) {
          throwException( "Missing dictionary information for the type of the pointer \"" +
                          m_objectType.cppName() + "\"",
                          "OraPtrReadBuffer::build" );
        }

        std::string ptrTypeName = ptrType.name();
        // Retrieve the relevant mapping element
        MappingElement::iterator iMe = m_mapping.find( ptrTypeName );
        if ( iMe == m_mapping.end() ) {
          throwException( "Item for \"" + ptrTypeName + "\" not found in the mapping element",
                          "OraPtrReadBuffer::build" );
        }
        RelationalStreamerFactory streamerFactory( m_schema );
        m_reader.reset( streamerFactory.newReader( ptrResolvedType, iMe->second ) );
        return m_reader->build( m_localElement, m_query );     
      }
      
      void* read( const std::vector<int>& fullId ){
        if(!m_dataElement) throwException( "Read buffer has not been built.","OraPtrReadBuffer::read");

        if(!fullId.size()) throwException( "Object id set is empty.","OraPtrReadBuffer::read"); 
        
        coral::AttributeList& whereBuff = m_query.whereData();
        coral::AttributeList::iterator iCol = whereBuff.begin();
        size_t i=0;
        for( coral::AttributeList::iterator iCol = whereBuff.begin(); iCol != whereBuff.end(); ++iCol ){
          if( i<fullId.size() ){
            iCol->data<int>() = fullId[i];
          }
          i++;
        }
        
        std::vector<int> recordId( fullId.size()-1 );
        for( size_t i=0; i<fullId.size()-1; i++ ){
          recordId[i] = fullId[i+1];
          ++i;
        }
        
        m_query.execute();
        m_reader->select( fullId[0] );
        void* destination = 0;
        if( m_query.nextCursorRow() ){
          destination = ClassUtils::constructObject( m_objectType.templateArgumentAt(0) );
          m_reader->setRecordId( recordId );
          m_reader->read( destination );
        }
        m_query.clear();
        m_reader->clear();
        return destination;
      }
      
    private:
      edm::TypeWithDict m_objectType;
      MappingElement& m_mapping;
      ContainerSchema& m_schema;
      DataElement m_localElement;
      SelectOperation m_query;
      DataElement* m_dataElement;
      std::auto_ptr<IRelationalReader> m_reader;
  };
    
  class RelationalPtrLoader: public IPtrLoader {
    public:
      RelationalPtrLoader( OraPtrReadBuffer& buffer, const std::vector<int>& fullId ):
        m_buffer( buffer ),
        m_fullId( fullId ),
        m_valid( true ){
      }
      
      virtual ~RelationalPtrLoader(){
      }
      
    public:
      void* load() const override {
        if(!m_valid){
          throwException("Ptr Loader has been invalidate.",
                         "RelationalPtrLoader::load");
        }
        return m_buffer.read( m_fullId );
      }

      void invalidate() override{
        m_valid = false;
      }

      bool isValid() const override{
        return m_valid;
      }
      
    private:
      OraPtrReadBuffer& m_buffer;
      std::vector<int> m_fullId;
      bool m_valid;
  };    
}

ora::OraPtrWriter::OraPtrWriter( const edm::TypeWithDict& objectType,
                                 MappingElement& mapping,
                                 ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_localElement(),
  m_dataElement( 0 ),
  m_writer(){
}
      
ora::OraPtrWriter::~OraPtrWriter(){
}

bool ora::OraPtrWriter::build(DataElement& dataElement,
                              IRelationalData& relationalData,
                              RelationalBuffer& operationBuffer){
  m_dataElement = &dataElement;
  m_localElement.clear();
  
  // Check the  type
  edm::TypeWithDict ptrType = m_objectType.templateArgumentAt(0);
  edm::TypeWithDict ptrResolvedType = ClassUtils::resolvedType(ptrType);
  // Check the component type
  if ( ! ptrType || !ptrResolvedType ) {
    throwException( "Missing dictionary information for the type of the pointer \"" +
                    m_objectType.cppName() + "\"",
                    "OraPtrWriter::build" );
  }

  std::string ptrTypeName = ptrType.name();
// Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( ptrTypeName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + ptrTypeName + "\" not found in the mapping element",
                    "OraPtrWriter::build" );
  }
  RelationalStreamerFactory streamerFactory( m_schema );
  m_writer.reset( streamerFactory.newWriter( ptrResolvedType, iMe->second ));
  return m_writer->build( m_localElement, relationalData, operationBuffer );
}

void ora::OraPtrWriter::setRecordId( const std::vector<int>& identity ){
  m_writer->setRecordId( identity );
}

/// Writes a data element
void ora::OraPtrWriter::write( int oid,
                               const void* data ){

  if(!m_dataElement){
    throwException("The streamer has not been built.",
                   "OraPtrWriter::write");    
  }
  
  edm::ObjectWithDict ptrObject( m_objectType, m_dataElement->address( data ) );
  // first load if required
  m_objectType.functionMemberByName("load").invoke(ptrObject,nullptr);
  // then get the data...
  void* ptrAddress = nullptr;
  edm::ObjectWithDict ptrAddrObj = edm::ObjectWithDict( edm::TypeWithDict(typeid(void*)), &ptrAddress );
  m_objectType.functionMemberByName("address").invoke(ptrObject, &ptrAddrObj);
  m_writer->write( oid, ptrAddress );
}

ora::OraPtrUpdater::OraPtrUpdater( const edm::TypeWithDict& objectType,
                                   MappingElement& mapping,
                                   ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ), 
  m_localElement(),
  m_dataElement( 0 ),
  m_updater(){
}
      
ora::OraPtrUpdater::~OraPtrUpdater(){
}

bool ora::OraPtrUpdater::build(DataElement& dataElement,
                               IRelationalData& relationalData,
                               RelationalBuffer& operationBuffer){
  m_dataElement = &dataElement;
  m_localElement.clear();
  
  // Check the  type
  edm::TypeWithDict ptrType = m_objectType.templateArgumentAt(0);
  edm::TypeWithDict ptrResolvedType = ClassUtils::resolvedType(ptrType);
  // Check the component type
  if ( ! ptrType || !ptrResolvedType ) {
    throwException( "Missing dictionary information for the type of the pointer \"" +
                    m_objectType.cppName() + "\"",
                    "OraPtrUpdater::build" );
  }

  std::string ptrTypeName = ptrType.name();
// Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( ptrTypeName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + ptrTypeName + "\" not found in the mapping element",
                    "OraPtrUpdater::build" );
  }
  RelationalStreamerFactory streamerFactory( m_schema );
  m_updater.reset( streamerFactory.newUpdater( ptrResolvedType, iMe->second ) );
  return m_updater->build( m_localElement, relationalData, operationBuffer );
}

void ora::OraPtrUpdater::setRecordId( const std::vector<int>& identity ){
  m_updater->setRecordId( identity );
}

/// Writes a data element
void ora::OraPtrUpdater::update( int oid,
                                 const void* data ){
  if(!m_dataElement){
    throwException("The streamer has not been built.",
                   "OraPtrUpdater::update");    
  }
  edm::ObjectWithDict ptrObject( m_objectType, m_dataElement->address( data ) );
  // first load if required
  m_objectType.functionMemberByName("load").invoke(ptrObject, nullptr);
  void *ptrAddress = nullptr;
  edm::ObjectWithDict ptrAddrObj = edm::ObjectWithDict( edm::TypeWithDict(typeid(void*)), &ptrAddress );
  m_objectType.functionMemberByName("address").invoke(ptrObject, &ptrAddrObj);
  m_updater->update( oid, ptrAddress );
}

ora::OraPtrReader::OraPtrReader( const edm::TypeWithDict& objectType,
                                 MappingElement& mapping,
                                 ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_dataElement( 0 ),
  m_readBuffer(),
  m_loaders(),
  m_tmpIds(){
  m_readBuffer.reset( new OraPtrReadBuffer( objectType, mapping, contSchema ));
}
      
ora::OraPtrReader::~OraPtrReader(){
  for(std::vector<boost::shared_ptr<IPtrLoader> >::const_iterator iL = m_loaders.begin();
      iL != m_loaders.end(); ++iL ){
    (*iL)->invalidate();
  }
}

bool ora::OraPtrReader::build( DataElement& dataElement,
                               IRelationalData& ){
  m_dataElement = &dataElement;
  m_tmpIds.clear();
  m_tmpIds.push_back(0);
  return m_readBuffer->build( dataElement );
}

void ora::OraPtrReader::select( int oid ){
  if(!m_dataElement) throwException( "The streamer has not been built.","OraPtrReader::select");
  m_tmpIds[0] = oid;
}

void ora::OraPtrReader::setRecordId( const std::vector<int>& identity ){
  m_tmpIds.resize( 1+identity.size() );
  for( size_t i=0;i<identity.size();i++){
    m_tmpIds[1+i] = identity[i];
  }
}

/// Read a data element
void ora::OraPtrReader::read( void* data ){
  if(!m_dataElement){
    throwException("The streamer has not been built.",
                   "OraPtrReader::read");    
  }
  // resolving loader address
  edm::MemberWithDict loaderMember = m_objectType.dataMemberByName("m_loader");
  DataElement& loaderElement = m_dataElement->addChild( loaderMember.offset(), 0 );
  void* loaderAddress = loaderElement.address( data );
  boost::shared_ptr<IPtrLoader>* loaderPtr = static_cast<boost::shared_ptr<IPtrLoader>*>( loaderAddress );
  // creating new loader
  boost::shared_ptr<IPtrLoader> newLoader( new RelationalPtrLoader( *m_readBuffer, m_tmpIds ) );
  m_loaders.push_back( newLoader );
  // installing the new loader
  *loaderPtr = newLoader;
}

void ora::OraPtrReader::clear(){
}


ora::OraPtrStreamer::OraPtrStreamer( const edm::TypeWithDict& objectType,
                                     MappingElement& mapping,
                                     ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::OraPtrStreamer::~OraPtrStreamer(){
}

ora::IRelationalWriter* ora::OraPtrStreamer::newWriter(){
  return new OraPtrWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::OraPtrStreamer::newUpdater(){
  return new OraPtrUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::OraPtrStreamer::newReader(){
  return new OraPtrReader( m_objectType, m_mapping, m_schema );
}

