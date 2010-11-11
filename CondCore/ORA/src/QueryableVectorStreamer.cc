#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Selection.h"
#include "CondCore/ORA/interface/QueryableVectorData.h"
#include "QueryableVectorStreamer.h"
#include "RelationalOperation.h"
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
#include "Reflex/Member.h"

namespace ora {

  class QVReader {
    public:
      QVReader( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema ):
        m_objectType( objectType ),
        m_mappingElement( mapping ),
        m_schema( contSchema ),
        m_recordId(),
        m_localElement(),
        m_query(),
        m_arrayHandler(),
        m_dataReader(),
        m_oid(-1){
      }

      ~QVReader(){
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

        Reflex::Type storeBaseType = ClassUtils::containerSubType(m_objectType,"store_base_type");
        if( !storeBaseType ){
          throwException( "Missing dictionary information for the store base type of the container \"" +
                          m_objectType.Name(Reflex::SCOPED) + "\"",
                          "QueryableVectorReadBuffer::build" );          
        }
        
        m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( storeBaseType ) );

        Reflex::Type valueType = ClassUtils::containerValueType(m_objectType);
        Reflex::Type valueResolvedType = ClassUtils::resolvedType(valueType);
        // Check the component type
        if ( ! valueType ||!valueResolvedType ) {
          throwException( "Missing dictionary information for the content type of the container \"" +
                          m_objectType.Name(Reflex::SCOPED) + "\"",
                          "QueryableVectorReadBuffer::build" );
        }
        std::string valueName = valueType.Name();
        // Retrieve the relevant mapping element
        MappingElement::iterator iMe = m_mappingElement.find( valueName );
        if ( iMe == m_mappingElement.end() ) {
          throwException( "Item for \"" + valueName + "\" not found in the mapping element",
                          "QueryableVectorReadBuffer::build" );
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
                              "QueryableVectorReadBuffer::setQueryCondition" );
            }
            MappingElement& valueTypeElement = iElem->second;
            if( valueTypeElement.elementType()==MappingElement::Primitive ){
              if(varName!="value_type"){
                throwException( "Item for element \"" + varName + "\" not found in the mapping element",
                                "QueryableVectorReadBuffer::setQueryCondition" );
              }
              colName = valueTypeElement.columnNames()[0];
            } else if( valueTypeElement.elementType()==MappingElement::Object ){
              MappingElement::iterator iInnerElem = valueTypeElement.find(varName);
              if ( iInnerElem == valueTypeElement.end() ) {
                throwException( "Item for element \"" + varName + "\" not found in the mapping element",
                                "QueryableVectorReadBuffer::setQueryCondition" );
              }
              colName = iInnerElem->second.columnNames()[0];
            } else {
              throwException( "Queries cannot be executed on types mapped on "+
                              MappingElement::elementTypeAsString(valueTypeElement.elementType()),
                              "QueryableVectorReadBuffer::setQueryCondition" );
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
      

      void select( const std::vector<int>& fullId ){
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
        m_query->execute();
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
      

      void read(void* address){

        if(!m_query.get()){
          throwException("The reader has not been built.",
                         "QVReader::read");
        }
        Reflex::Type iteratorDereferenceReturnType = m_arrayHandler->iteratorReturnType();
        Reflex::Member firstMember = iteratorDereferenceReturnType.MemberByName( "first" );
        if ( ! firstMember ) {
          throwException( "Could not retrieve the data member \"first\" of the class \"" +
                          iteratorDereferenceReturnType.Name(Reflex::SCOPED) + "\"",
                          "QueryableVectorReadBuffer::read" );
        }
        Reflex::Member secondMember = iteratorDereferenceReturnType.MemberByName( "second" );
        if ( ! secondMember ) {
          throwException( "Could not retrieve the data member \"second\" of the class \"" +
                          iteratorDereferenceReturnType.Name(Reflex::SCOPED) + "\"",
                          "QueryableVectorReadBuffer::read" );
        }

        m_arrayHandler->clear( address );

        unsigned int i=0;
        while ( m_query->nextCursorRow() ){

          // Create a new element for the array
          void* objectData = iteratorDereferenceReturnType.Construct().Address();
          void* positionData = static_cast< char* >( objectData ) + firstMember.Offset();
          void* containerData = static_cast< char* >( objectData ) + secondMember.Offset();

          m_recordId[m_recordId.size()-1] = (int)i;
          coral::AttributeList& row = m_query->data();

          *(size_t*)positionData = (size_t)(row[m_mappingElement.posColumn()].data<int>());
    
          m_dataReader->setRecordId( m_recordId );
          m_dataReader->select( m_oid );
          m_dataReader->read( containerData );

          size_t prevSize = m_arrayHandler->size( address );
          m_arrayHandler->appendNewElement( address, objectData );
          bool inserted = m_arrayHandler->size( address )>prevSize;
          
          iteratorDereferenceReturnType.Destruct( objectData );
          if ( !inserted ) {
            throwException( "Could not insert a new element in the array type \"" +
                            m_objectType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                            "QueryableVectorReadBuffer::read" );
          }
          ++i;
        }

        m_arrayHandler->finalize( address );
      }

    private:
      Reflex::Type m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;
      std::auto_ptr<SelectOperation> m_query;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalReader> m_dataReader;
      int m_oid;
  };  

  class RelationalVectorLoader: public IVectorLoader {

      public:

        // constructor
      RelationalVectorLoader( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema,
                              const std::vector<int>& fullId ):
        m_isValid(true),
        m_reader( objectType, mapping, contSchema ),
        m_identity(fullId){
      }

      // destructor
      virtual ~RelationalVectorLoader(){
      }

      public:

      // triggers the data loading
      bool load(void* address) const {
        bool ret = false;
        if(m_isValid) {
          m_reader.build();
          m_reader.select( m_identity );
          m_reader.read( address );
          ret = true;
        }
        return ret;
      }

      bool loadSelection(const Selection& selection, void* address) const {
        bool ret = false;
        if(m_isValid) {
          m_reader.build();
          m_reader.select( m_identity, selection );
          m_reader.read( address );
          ret = true;
        }
        return ret;
      }

      size_t getSelectionCount( const Selection& selection ) const {
        size_t ret = 0;
        if(m_isValid) {
          ret = m_reader.selectionCount( m_identity, selection );
        }
        return ret;
      }

      void invalidate(){
        m_isValid = false;
      }

      bool isValid() const{
        return m_isValid;
      }
        
      private:
        bool m_isValid;
        mutable QVReader m_reader;
        std::vector<int> m_identity;
    };
  
}

ora::QueryableVectorWriter::QueryableVectorWriter( const Reflex::Type& objectType,
                                                   MappingElement& mapping,
                                                   ContainerSchema& contSchema ):
  m_objectType( objectType ),
  m_mappingElement( mapping ),
  m_schema( contSchema ),
  m_recordId(),
  m_localElement(),
  m_offset(0),
  m_insertOperation( 0 ),
  m_arrayHandler(){
}

  ora::QueryableVectorWriter::~QueryableVectorWriter(){
}

bool ora::QueryableVectorWriter::build( DataElement& offset,
                                        IRelationalData& data,
                                        RelationalBuffer& operationBuffer ){
  m_localElement.clear();
  m_recordId.clear();
  // allocate for the index...
  m_recordId.push_back(0);
    
  RelationalStreamerFactory streamerFactory( m_schema );
  
  // first open the insert on the extra table...
  m_insertOperation = &operationBuffer.newBulkInsert( m_mappingElement.tableName() );
  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( !columns.size() ){
    throwException( "Id columns not found in the mapping.",
                    "QueryableVectorWriter::build");    
  }
  for( size_t i=0; i<columns.size(); i++ ){
    m_insertOperation->addId( columns[ i ] );
  }

  m_offset = &offset;

  Reflex::Type storeBaseType = ClassUtils::containerSubType(m_objectType,"store_base_type");
  if( !storeBaseType ){
    throwException( "Missing dictionary information for the store base type of the container \"" +
                    m_objectType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                    "QueryableVectorWriter::build" );    
  }
  
  m_arrayHandler.reset( ArrayHandlerFactory::newArrayHandler( storeBaseType ) );
  
  Reflex::Type valueType = ClassUtils::containerValueType(m_objectType);
  Reflex::Type valueResolvedType = ClassUtils::resolvedType(valueType);
  // Check the component type
  if ( ! valueType || !valueResolvedType ) {
    throwException( "Missing dictionary information for the content type of the container \"" +
                    m_objectType.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                    "QueryableVectorWriter::build" );
  }
  
  std::string valueName = valueType.Name();
  // Retrieve the relevant mapping element
  MappingElement::iterator iMe = m_mappingElement.find( valueName );
  if ( iMe == m_mappingElement.end() ) {
    throwException( "Item for \"" + valueName + "\" not found in the mapping element",
                    "QueryableVectorWriter::build" );
  }

  m_dataWriter.reset( streamerFactory.newWriter( valueResolvedType, iMe->second ) );
  m_dataWriter->build( m_localElement, *m_insertOperation, operationBuffer );
  return true;
}

  void ora::QueryableVectorWriter::setRecordId( const std::vector<int>& identity ){
  m_recordId.clear();
  for(size_t i=0;i<identity.size();i++) {
    m_recordId.push_back( identity[i] );
  }
  m_recordId.push_back( 0 );
}
      
void ora::QueryableVectorWriter::write( int oid,
                                        const void* inputData ){

  if(!m_offset){
    throwException("The streamer has not been built.",
                   "QueryableVectorWriter::write");
  }

  const std::vector<std::string>& columns = m_mappingElement.columnNames();
  if( columns.size() != m_recordId.size()+1){
    throwException( "Object id elements provided are not matching with the mapped id columns.",
                    "QueryableVectorWriter::write");
  }
  
  void* vectorAddress = m_offset->address( inputData );
  Reflex::Object vectorObj( m_objectType,const_cast<void*>(vectorAddress));
  vectorObj.Invoke("load",0);
  void* storageAddress = 0;
  vectorObj.Invoke("storageAddress",storageAddress);
  
  // Use the iterator to loop over the elements of the container.
  size_t containerSize = m_arrayHandler->size( storageAddress  );
  size_t persistentSize = m_arrayHandler->persistentSize( storageAddress  );

  if ( containerSize == 0 || containerSize < persistentSize ) return;
  if ( containerSize > MAXARRAYSIZE ){
    std::stringstream ms;
    ms << "Cannot store non-blob array with size>" << MAXARRAYSIZE;
    throwException( ms.str(),
                    "QueryableVectorWriter::write" );    
  }

  size_t startElementIndex = m_arrayHandler->startElementIndex( storageAddress );

  std::auto_ptr<IArrayIteratorHandler> iteratorHandler( m_arrayHandler->iterate( storageAddress ) );
  const Reflex::Type& iteratorDereferenceReturnType = iteratorHandler->returnType();
  Reflex::Member secondMember = iteratorDereferenceReturnType.MemberByName( "second" );
  if ( ! secondMember ) {
    throwException( "Could not retrieve the data member \"second\" for the class \"" +
                    iteratorDereferenceReturnType.Name(Reflex::SCOPED) + "\"",
                    "QueryableVectorWriter::write" );
  }

  coral::IBulkOperation& bulkInsert = m_insertOperation->setUp( containerSize-startElementIndex+1 );

  for ( size_t iIndex = startElementIndex; iIndex < containerSize; ++iIndex ) {

    m_recordId[m_recordId.size()-1] = iIndex;
    coral::AttributeList& dataBuff = m_insertOperation->data();

    dataBuff[ columns[0] ].data<int>() = oid;
    for( size_t i = 1;i < columns.size(); i++ ){
      dataBuff[ columns[i] ].data<int>() = m_recordId[i-1];
    }


    void* objectReference = iteratorHandler->object();
    void* componentData = static_cast< char* >( objectReference ) + secondMember.Offset();

    m_dataWriter->setRecordId( m_recordId );
    m_dataWriter->write( oid, componentData );

    bulkInsert.processNextIteration();
    
    // Increment the iterator
    iteratorHandler->increment();
  }

  // execute the insert...
  m_arrayHandler->finalize( const_cast<void*>( storageAddress ) );
}

ora::QueryableVectorUpdater::QueryableVectorUpdater(const Reflex::Type& objectType,
                                                    MappingElement& mapping,
                                                    ContainerSchema& contSchema ):
  m_buffer( 0 ),
  m_writer( objectType, mapping, contSchema ){
}

ora::QueryableVectorUpdater::~QueryableVectorUpdater(){
}

bool ora::QueryableVectorUpdater::build( DataElement& offset,
                                         IRelationalData& relationalData,
                                         RelationalBuffer& operationBuffer){
  m_buffer = &operationBuffer;
  return m_writer.build( offset, relationalData, operationBuffer );
}

void ora::QueryableVectorUpdater::setRecordId( const std::vector<int>& identity ){
  m_writer.setRecordId( identity );
}

void ora::QueryableVectorUpdater::update( int oid,
                                          const void* data ){
  if( !m_writer.dataElement() ){
    throwException("The streamer has not been built.",
                   "QueryableVectorUpdater::update");
  }
  
  void* vectorAddress = m_writer.dataElement()->address( data );
  Reflex::Object vectorObj( m_writer.objectType(),const_cast<void*>(vectorAddress));
  vectorObj.Invoke("load",0);
  void* storageAddress = 0;
  vectorObj.Invoke("storageAddress",storageAddress);
  
  IArrayHandler& arrayHandler = *m_writer.arrayHandler();
  
  size_t arraySize = arrayHandler.size(storageAddress);
  size_t persistentSize = arrayHandler.persistentSize(storageAddress);
  if(persistentSize>arraySize){
    deleteArrayElements( m_writer.mapping(), oid, arraySize, *m_buffer );
  }
  m_writer.write( oid, data );
}
  

  

ora::QueryableVectorReader::QueryableVectorReader(const Reflex::Type& objectType,
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

  void* arrayAddress = m_dataElement->address( destinationData );

  boost::shared_ptr<IVectorLoader> loader( new RelationalVectorLoader( m_objectType, m_mapping, m_schema, m_tmpIds ) );
  m_loaders.push_back(loader);

  LoaderClient* client = static_cast<LoaderClient*>( arrayAddress );
  client->install( loader );  
}

void ora::QueryableVectorReader::clear(){
}


ora::QueryableVectorStreamer::QueryableVectorStreamer( const Reflex::Type& objectType,
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
