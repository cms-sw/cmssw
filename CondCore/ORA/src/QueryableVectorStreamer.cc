#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Selection.h"
#include "CondCore/ORA/interface/QueryableVector.h"
#include "QueryableVectorStreamer.h"
#include "MultiRecordInsertOperation.h"
#include "RelationalStreamerFactory.h"
#include "MappingElement.h"
#include "ContainerSchema.h"
#include "ArrayHandlerFactory.h"
#include "ClassUtils.h"
#include "IArrayHandler.h"
#include "ArrayCommonImpl.h"
#include "RelationalBuffer.h"
// externals
#include "RelationalAccess/IBulkOperation.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "CoralBase/Attribute.h"

namespace ora {

  class QVReader {
    public:
      QVReader( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema ):
        m_objectType( objectType ),
        m_localElement(),
        m_reader( ClassUtils::containerSubType(objectType,"store_base_type"), mapping, contSchema ){
        m_localElement.clear();
        InputRelationalData dummy;
        m_reader.build( m_localElement,dummy ); 
      }

    void read( const std::vector<int>& fullId, void* destinationAddress){
      m_reader.select( fullId[0] );
      std::vector<int> recordId;
      for(size_t i=1;i<fullId.size();i++) {
         recordId.push_back( fullId[i] );
      }
      m_reader.setRecordId(recordId);
      m_reader.read( destinationAddress );
      m_reader.clear();
    } 

    private:
      edm::TypeWithDict m_objectType;
      DataElement m_localElement;
      PVectorReader m_reader;
  };

  class QVQueryMaker {
    public:
      QVQueryMaker( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema ):
        m_objectType( objectType ),
        m_mappingElement( mapping ),
        m_schema( contSchema ),
        m_recordId(),
        m_localElement(),
        m_query(),
        m_arrayHandler(),
        m_oid(-1){
      }

      ~QVQueryMaker(){
      }
      
      bool build(){
        m_localElement.clear();
        m_recordId.clear();
        // allocate for the index...
        m_recordId.push_back(0);

        RelationalStreamerFactory streamerFactory( m_schema );
        // first open the insert on the extra table...
        m_query.reset( new SelectOperation( m_mappingElement.tableName(), m_schema.storageSchema() ));

        m_query->addWhereId( m_mappingElement.pkColumn() );
        std::vector<std::string> recIdCols = m_mappingElement.recordIdColumns();
        for( size_t i=0; i<recIdCols.size(); i++ ){
          m_query->addId( recIdCols[ i ] );
          m_query->addOrderId( recIdCols[ i ] );
        }

        edm::TypeWithDict storeBaseType = ClassUtils::containerSubType(m_objectType,"range_store_base_type");
        if( !storeBaseType ){
          throwException( "Missing dictionary information for the range store base type of the container \"" +
                          m_objectType.cppName() + "\"",
                          "QVQueryMaker::build" );          
        }
        
        m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( storeBaseType ) );

        edm::TypeWithDict valueType = ClassUtils::containerValueType(m_objectType);
        edm::TypeWithDict valueResolvedType = ClassUtils::resolvedType(valueType);
        // Check the component type
        if ( ! valueType ||!valueResolvedType ) {
          throwException( "Missing dictionary information for the content type of the container \"" +
                          m_objectType.cppName() + "\"",
                          "QVQueryMaker::build" );
        }
        // Retrieve the relevant mapping element
        MappingElement::iterator iMe = m_mappingElement.find( "value_type" );
        if ( iMe == m_mappingElement.end() ) {
          throwException( "Item for \"value_type\" not found in the mapping element",
                          "QVQueryMaker::build" );
        }

        m_dataReader.reset( streamerFactory.newReader( valueResolvedType, iMe->second ) );
        m_dataReader->build( m_localElement, *m_query );
        return true;
      }

      void setQueryCondition( IRelationalData& queryData, const Selection& selection, MappingElement& mappingElement ){
        coral::AttributeList& whereData = queryData.whereData();
        // adding the selection conditions
        const std::vector<std::pair<std::string,std::string> >& theItems =  selection.items();
        std::stringstream cond;
        unsigned int i=0;
        for(std::vector<std::pair<std::string,std::string> >::const_iterator iItem = theItems.begin();
            iItem != theItems.end();
            ++iItem){
          cond << " AND ";
          std::string varName = Selection::variableNameFromUniqueString(iItem->first);
          std::stringstream selColumn;
          std::string colName("");
          if(varName == Selection::indexVariable()){
            colName = mappingElement.columnNames()[mappingElement.columnNames().size()-1]; // the position column is the last
            selColumn << colName<<"_"<<i;
            whereData.extend<int>(selColumn.str());
            whereData[selColumn.str()].data<int>() = selection.data()[iItem->first].data<int>();
          } else {
            MappingElement::iterator iElem = mappingElement.find("value_type");
            if ( iElem == mappingElement.end() ) {
              throwException( "Item for element \"value_type\" not found in the mapping element",
                              "QVQueryMaker::setQueryCondition" );
            }
            MappingElement& valueTypeElement = iElem->second;
            if( valueTypeElement.elementType()==MappingElement::Primitive ){
              if(varName!="value_type"){
                throwException( "Item for element \"" + varName + "\" not found in the mapping element",
                                "QVQueryMaker::setQueryCondition" );
              }
              colName = valueTypeElement.columnNames()[0];
            } else if( valueTypeElement.elementType()==MappingElement::Object ){
              MappingElement::iterator iInnerElem = valueTypeElement.find(varName);
              if ( iInnerElem == valueTypeElement.end() ) {
                throwException( "Item for element \"" + varName + "\" not found in the mapping element",
                                "QVQueryMaker::setQueryCondition" );
              }
              colName = iInnerElem->second.columnNames()[0];
            } else {
              throwException( "Queries cannot be executed on types mapped on "+
                              MappingElement::elementTypeAsString(valueTypeElement.elementType()),
                              "QVQueryMaker::setQueryCondition" );
            }
            selColumn << colName<<"_"<<i;
            whereData.extend(selColumn.str(),selection.data()[iItem->first].specification().type());
            whereData[selColumn.str()].setValueFromAddress(selection.data()[iItem->first].addressOfData());
          }
          cond << colName << " " << iItem->second << " :"<<selColumn.str();
          i++;
          selColumn.str("");
        }

        // add the resulting condition clause
        queryData.whereClause()+=cond.str();
      }
      
      void select( const std::vector<int>& fullId, const Selection& selection ){
        if(!m_query.get()){
          throwException("The reader has not been built.",
                         "QVReader::select");
        }
        
        m_oid = fullId[0];
        m_recordId.clear();
        for(size_t i=1;i<fullId.size();i++) {
          m_recordId.push_back( fullId[i] );
        }
        // allocate the element for the index...
        m_recordId.push_back( 0 );

        coral::AttributeList& whereData = m_query->whereData();
        whereData[ m_mappingElement.pkColumn() ].data<int>() = fullId[0];

        setQueryCondition( *m_query, selection, m_mappingElement );
        
        m_query->execute();
      }

      size_t selectionCount( const std::vector<int>& fullId, const Selection& selection ){
        SelectOperation countQuery( m_mappingElement.tableName(), m_schema.storageSchema() );
        std::string countColumn("COUNT(*)");
        countQuery.addData( countColumn ,typeid(int) );
        countQuery.addWhereId( m_mappingElement.pkColumn() );
        std::vector<std::string> recIdColumns = m_mappingElement.recordIdColumns();
        for( size_t i=0;i<recIdColumns.size();i++){
          countQuery.addWhereId( recIdColumns[i] );
        }

        coral::AttributeList& whereData = countQuery.whereData();
        // Fill-in the identities.
        whereData[ m_mappingElement.pkColumn() ].data<int>() = fullId[0];
        for ( size_t i=0;i<fullId.size();i++ ){
          whereData[ recIdColumns[i] ].data<int>() = fullId[i+1];
        }
        
        setQueryCondition( countQuery, selection, m_mappingElement );
        countQuery.execute();

        size_t result = 0;
        if( countQuery.nextCursorRow() ){
          coral::AttributeList& row = countQuery.data();
          result = row[countColumn].data<int>();
        }
        return result;
      }
      
      void executeAndLoad(void* address){

        if(!m_query.get()){
          throwException("The reader has not been built.",
                         "QVReader::read");
        }
        edm::TypeWithDict iteratorDereferenceReturnType = m_arrayHandler->iteratorReturnType();
        edm::MemberWithDict firstMember = iteratorDereferenceReturnType.dataMemberByName( "first" );
        if ( ! firstMember ) {
          throwException( "Could not retrieve the data member \"first\" of the class \"" +
                          iteratorDereferenceReturnType.cppName() + "\"",
                          "QVQueryMakerAndLoad::read" );
        }
        edm::MemberWithDict secondMember = iteratorDereferenceReturnType.dataMemberByName( "second" );
        if ( ! secondMember ) {
          throwException( "Could not retrieve the data member \"second\" of the class \"" +
                          iteratorDereferenceReturnType.cppName() + "\"",
                          "QVQueryMakerAndLoad::read" );
        }

        m_arrayHandler->clear( address );

        unsigned int i=0;
        while ( m_query->nextCursorRow() ){

          // Create a new element for the array
          void* objectData = iteratorDereferenceReturnType.construct().address();
          void* positionData  = static_cast< char* >( objectData ) + firstMember.offset();
          void* containerData = static_cast< char* >( objectData ) + secondMember.offset();

          m_recordId[m_recordId.size()-1] = (int)i;
          coral::AttributeList& row = m_query->data();

          *(size_t*)positionData = (size_t)(row[m_mappingElement.posColumn()].data<int>());
    
          m_dataReader->setRecordId( m_recordId );
          m_dataReader->select( m_oid );
          m_dataReader->read( containerData );

          size_t prevSize = m_arrayHandler->size( address );
          m_arrayHandler->appendNewElement( address, objectData );
          bool inserted = m_arrayHandler->size( address )>prevSize;
          
          iteratorDereferenceReturnType.destruct( objectData );
          if ( !inserted ) {
            throwException( "Could not insert a new element in the array type \"" +
                            m_objectType.cppName() + "\"",
                            "QVQueryMakerAndLoad::executeAndLoad" );
          }
          ++i;
        }

        m_arrayHandler->finalize( address );
	m_query->clear();
      }

    private:
      edm::TypeWithDict m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;
      std::auto_ptr<SelectOperation> m_query;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalReader> m_dataReader;
      int m_oid;
  };  

  class QueryableVectorLoader: public IVectorLoader {

      public:

        // constructor
      QueryableVectorLoader( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema,
                              const std::vector<int>& fullId ):
        m_isValid(true),
        m_reader( objectType, mapping, contSchema ),
        m_queryMaker( objectType, mapping, contSchema ),
        m_identity(fullId){
      }

      // destructor
      virtual ~QueryableVectorLoader(){
      }

      public:

      // triggers the data loading
      bool load(void* address) const override {
        bool ret = false;
        if(m_isValid) {
          m_reader.read( m_identity, address );
          ret = true;
        }
        return ret;
      }

      bool loadSelection(const Selection& selection, void* address) const override {
        bool ret = false;
        if(m_isValid) {
          m_queryMaker.build();
          m_queryMaker.select( m_identity, selection );
          m_queryMaker.executeAndLoad( address );
          ret = true;
        }
        return ret;
      }

      size_t getSelectionCount( const Selection& selection ) const override {
        size_t ret = 0;
        if(m_isValid) {
          ret = m_queryMaker.selectionCount( m_identity, selection );
        }
        return ret;
      }

      void invalidate() override{
        m_isValid = false;
      }

      bool isValid() const override{
        return m_isValid;
      }
        
      private:
        bool m_isValid;
        mutable QVReader m_reader;
        mutable QVQueryMaker m_queryMaker;
        std::vector<int> m_identity;
    };
  
}

ora::QueryableVectorWriter::QueryableVectorWriter( const edm::TypeWithDict& objectType,
                                                   MappingElement& mapping,
                                                   ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_offset( 0 ),
  m_localElement(),
  m_writer(ClassUtils::containerSubType(objectType,"store_base_type"), mapping, contSchema ){
}

ora::QueryableVectorWriter::~QueryableVectorWriter(){
}

bool ora::QueryableVectorWriter::build( DataElement& offset,
                                        IRelationalData& data,
                                        RelationalBuffer& operationBuffer ){
  m_offset = &offset;
  m_localElement.clear();
  return m_writer.build( m_localElement, data, operationBuffer );
}

void ora::QueryableVectorWriter::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}
      
void ora::QueryableVectorWriter::write( int oid,
                                        const void* inputData ){
  if(!m_offset){
    throwException("The streamer has not been built.",
                   "QueryableVectorWriter::write");
  }
  void* vectorAddress = m_offset->address( inputData );
  edm::ObjectWithDict vectorObj( m_objectType, const_cast<void*>(vectorAddress) );
  m_objectType.functionMemberByName("load").invoke(vectorObj,nullptr);
  void* storageAddress = nullptr;
  edm::ObjectWithDict storAddObj = edm::ObjectWithDict( edm::TypeWithDict(typeid(void*)), &storageAddress );
  m_objectType.functionMemberByName("storageAddress").invoke(vectorObj, &storAddObj);
  m_writer.write( oid, storageAddress );
}

ora::QueryableVectorUpdater::QueryableVectorUpdater(const edm::TypeWithDict& objectType,
                                                    MappingElement& mapping,
                                                    ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_offset( 0 ),
  m_localElement(),
  m_updater( ClassUtils::containerSubType(objectType,"store_base_type"), mapping, contSchema ){
}

ora::QueryableVectorUpdater::~QueryableVectorUpdater(){
}

bool ora::QueryableVectorUpdater::build( DataElement& offset,
                                         IRelationalData& relationalData,
                                         RelationalBuffer& operationBuffer){
  m_offset = &offset;
  m_localElement.clear();
  return m_updater.build( m_localElement, relationalData, operationBuffer );
}

void ora::QueryableVectorUpdater::setRecordId( const std::vector<int>& identity ){
  m_updater.setRecordId( identity );
}

void ora::QueryableVectorUpdater::update( int oid,
                                          const void* data ){
  if(!m_offset){
    throwException("The streamer has not been built.",
                   "QueryableVectorUpdater::update");
  }
  void* vectorAddress = m_offset->address( data );
  edm::ObjectWithDict vectorObj( m_objectType,const_cast<void*>(vectorAddress));
  m_objectType.functionMemberByName("load").invoke(vectorObj, nullptr);
  void* storageAddress = nullptr;
  edm::ObjectWithDict storAddObj = edm::ObjectWithDict( edm::TypeWithDict(typeid(void*)), &storageAddress );
  m_objectType.functionMemberByName("storageAddress").invoke(vectorObj, &storAddObj);
  m_updater.update( oid, storageAddress );
}  
  
ora::QueryableVectorReader::QueryableVectorReader(const edm::TypeWithDict& objectType,
                                                  MappingElement& mapping,
                                                  ContainerSchema& contSchema ):
  m_objectType(objectType),
  m_mapping( mapping ),
  m_schema( contSchema ),
  m_dataElement( 0 ),
  m_loaders(),
  m_tmpIds(){
}

ora::QueryableVectorReader::~QueryableVectorReader(){
  for(std::vector<boost::shared_ptr<IVectorLoader> >::const_iterator iL = m_loaders.begin();
      iL != m_loaders.end(); ++iL ){
    (*iL)->invalidate();
  }
}

bool ora::QueryableVectorReader::build( DataElement& dataElement,
                                        IRelationalData& ){
  m_dataElement = &dataElement;
  m_tmpIds.clear();
  m_tmpIds.push_back(0);
  return true;
}

void ora::QueryableVectorReader::select( int oid ){
  m_tmpIds[0] = oid;
}

void ora::QueryableVectorReader::setRecordId( const std::vector<int>& identity ){
  m_tmpIds.resize( 1+identity.size() );
  for( size_t i=0;i<identity.size();i++){
    m_tmpIds[1+i] = identity[i];
  }
}

void ora::QueryableVectorReader::read( void* destinationData ) {
  if(!m_dataElement){
    throwException("The streamer has not been built.",
                   "QueryableVectorReader::read");
  }

  // finding the address (on the instance) where to install the loader
  edm::MemberWithDict loaderMember = m_objectType.dataMemberByName("m_loader");
  if(!loaderMember){
    throwException("The loader member has not been found.",
                   "QueryableVectorReader::read");     
  }
  DataElement& loaderMemberElement = m_dataElement->addChild( loaderMember.offset(), 0 );
  void* loaderAddress = loaderMemberElement.address( destinationData );

  // creating and registering the new loader to assign
  boost::shared_ptr<IVectorLoader> loader( new QueryableVectorLoader( m_objectType, m_mapping, m_schema, m_tmpIds ) );
  m_loaders.push_back(loader);
  // installing the loader
  *(static_cast<boost::shared_ptr<IVectorLoader>*>( loaderAddress )) = loader;
}

void ora::QueryableVectorReader::clear(){
}


ora::QueryableVectorStreamer::QueryableVectorStreamer( const edm::TypeWithDict& objectType,
                                                       MappingElement& mapping,
                                                       ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mapping( mapping ),
  m_schema( contSchema ){
}

ora::QueryableVectorStreamer::~QueryableVectorStreamer(){
}

ora::IRelationalWriter* ora::QueryableVectorStreamer::newWriter(){
  return new QueryableVectorWriter( m_objectType, m_mapping, m_schema );
}

ora::IRelationalUpdater* ora::QueryableVectorStreamer::newUpdater(){
  return new QueryableVectorUpdater( m_objectType, m_mapping, m_schema );
}

ora::IRelationalReader* ora::QueryableVectorStreamer::newReader(){
  return new QueryableVectorReader( m_objectType, m_mapping, m_schema );
}
