#include "CondCore/ORA/interface/Exception.h"
#include "PoolDatabaseSchema.h"
#include "MappingRules.h"
#include "MappingElement.h"
//
#include <memory>
// externals
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"
#include "CoralBase/Attribute.h"

std::string ora::PoolMainTable::tableName(){
  static std::string s_name("POOL_RSS_DB");
  return s_name;
}

ora::PoolMainTable::PoolMainTable( coral::ISchema& dbSchema ):
  m_schema( dbSchema ){
}

ora::PoolMainTable::~PoolMainTable(){
}

bool ora::PoolMainTable::getParameters( std::map<std::string,std::string>& ){
  return false;
}

std::string ora::PoolMainTable::schemaVersion(){
  return poolSchemaVersion();
}

bool ora::PoolMainTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::PoolMainTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "POOL database main table already exists in this schema.",
                    "PoolMainTable::create");
  }
  throwException( "POOL database cannot be created.","PoolMainTable::create");
}

void ora::PoolMainTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::PoolSequenceTable::tableName(){
  static std::string s_name("POOL_RSS_SEQ");
  return s_name;
}

std::string ora::PoolSequenceTable::sequenceNameColumn(){
  static std::string s_column("NAME");
  return s_column;
}

std::string ora::PoolSequenceTable::sequenceValueColumn(){
  static std::string s_column("VALUE");
  return s_column;
}

ora::PoolSequenceTable::PoolSequenceTable( coral::ISchema& schema ):
  m_schema( schema),
  m_dbCache( 0 ){
}

ora::PoolSequenceTable::~PoolSequenceTable(){
}

void ora::PoolSequenceTable::init( PoolDbCache& dbCache ){
  m_dbCache = &dbCache;
}

bool
ora::PoolSequenceTable::add( const std::string& sequenceName ){
  // Create the entry in the table if it does not exist.
  coral::AttributeList insertData;
  insertData.extend<std::string>(sequenceNameColumn());
  insertData.extend<int>(sequenceValueColumn());
  coral::AttributeList::iterator iAttribute = insertData.begin();
  iAttribute->data< std::string >() = sequenceName;
  ++iAttribute;
  iAttribute->data< int >() = 0;
  m_schema.tableHandle( tableName() ).dataEditor().insertRow( insertData );
  return true;
}

bool
ora::PoolSequenceTable::getLastId( const std::string& sequenceName,
                                   int& lastId ) {
  if(!m_dbCache){
    throwException("Sequence Table handle has not been initialized.","PoolSequenceTable::getLastId");
  }
  
  // first lookup in the cache for the built in sequences...
  std::map<std::string,PoolDbCacheData*>& seq = m_dbCache->sequences();
  std::map<std::string,PoolDbCacheData*>::iterator iS = seq.find( sequenceName );
  if( iS != seq.end()){
    if( iS->second->m_nobjWr == 0 ) return false;
    lastId = iS->second->m_nobjWr-1;
    return true;
  }
  
  // otherwise, look up into the regular sequence table.
  std::auto_ptr< coral::IQuery > query( m_schema.tableHandle( tableName() ).newQuery() );
  query->limitReturnedRows( 1, 0 );
  query->addToOutputList( sequenceValueColumn() );
  query->defineOutputType( sequenceValueColumn(), coral::AttributeSpecification::typeNameForType<int>() );
  query->setForUpdate();
  std::string whereClause( sequenceNameColumn() + " = :" + sequenceNameColumn() );
  coral::AttributeList rowData;
  rowData.extend<std::string>(sequenceNameColumn());
  rowData.begin()->data< std::string >() = sequenceName;
  query->setCondition( whereClause, rowData );
  coral::ICursor& cursor = query->execute();
  if ( cursor.next() ) {
    lastId = cursor.currentRow().begin()->data<int >();
    return true;
  }
  return false;
}

void ora::PoolSequenceTable::sinchronize( const std::string& sequenceName,
                                          int lastValue ){
  if(!m_dbCache){
    throwException("Sequence Table handle has not been initialized.","PoolSequenceTable::sinchronize");
  }
  // nothing to do in the db if the sequence is in the cache...
  std::map<std::string,PoolDbCacheData*>& seq = m_dbCache->sequences();
  std::map<std::string,PoolDbCacheData*>::iterator iS = seq.find( sequenceName );
  if( iS != seq.end()){
    iS->second->m_nobjWr = lastValue+1;
    return;
  }

  coral::AttributeList updateData;
  updateData.extend<std::string>(sequenceNameColumn());
  updateData.extend<int>(sequenceValueColumn());
  std::string setClause( sequenceValueColumn() + " = :" +  sequenceValueColumn() );
  std::string whereClause( sequenceNameColumn() + " = :" + sequenceNameColumn() );
  // Increment the oid in the database as well
  coral::AttributeList::iterator iAttribute = updateData.begin();
  iAttribute->data< std::string >() = sequenceName;
  ++iAttribute;
  iAttribute->data< int >() = lastValue;
  m_schema.tableHandle( tableName() ).dataEditor().updateRows( setClause,whereClause,updateData );
}

void ora::PoolSequenceTable::erase( const std::string& sequenceName ){
  coral::AttributeList whereData;
  whereData.extend<std::string>(sequenceNameColumn());
  whereData[ sequenceNameColumn() ].data<std::string>() = sequenceName;
  std::string whereClause( sequenceNameColumn() + " = :" + sequenceNameColumn() );
  m_schema.tableHandle( tableName() ).dataEditor().deleteRows( whereClause, whereData );
}

bool ora::PoolSequenceTable::exists(){
  if(!m_dbCache){
    throwException("Sequence Table handle has not been initialized.","PoolSequenceTable::exists");
  }
  // ????
  return m_schema.existsTable( tableName() );
}

void ora::PoolSequenceTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "POOL database sequence table already exists in this schema.",
                    "PoolSequenceTable::create");
  }
  throwException( "POOL database cannot be created.","PoolSequenceTable::create");  
}

void ora::PoolSequenceTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::PoolMappingVersionTable::tableName(){
  static std::string s_table("POOL_OR_MAPPING_VERSIONS");
  return s_table;
}

std::string ora::PoolMappingVersionTable::mappingVersionColumn(){
  static std::string s_col("MAPPING_VERSION");
  return s_col;
}

std::string ora::PoolMappingVersionTable::containerNameColumn(){
  static std::string s_col("CONTAINER_ID");
  return s_col;
}

ora::PoolMappingVersionTable::PoolMappingVersionTable( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::PoolMappingVersionTable::~PoolMappingVersionTable(){
}

bool ora::PoolMappingVersionTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::PoolMappingVersionTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "POOL database mapping version table already exists in this schema.",
                    "PoolMappingVersionTable::create");
  }
  throwException( "POOL database cannot be created.","PoolMappingVersionTable::create");  
}

void ora::PoolMappingVersionTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}


std::string ora::PoolMappingElementTable::tableName(){
  static std::string s_table("POOL_OR_MAPPING_ELEMENTS");
  return s_table;  
}

std::string ora::PoolMappingElementTable::mappingVersionColumn(){
  static std::string s_col("MAPPING_VERSION");
  return s_col;  
}

std::string ora::PoolMappingElementTable::elementIdColumn(){
  static std::string s_col("ELEMENT_ID");
  return s_col;
}

std::string ora::PoolMappingElementTable::elementTypeColumn(){
  static std::string s_col("ELEMENT_TYPE");
  return s_col;  
}

std::string ora::PoolMappingElementTable::scopeNameColumn(){
  static std::string s_col("VARIABLE_SCOPE");
  return s_col;
}

std::string ora::PoolMappingElementTable::variableNameColumn(){
  static std::string s_col("VARIABLE_NAME");
  return s_col;
}

std::string ora::PoolMappingElementTable::variableParIndexColumn(){
  static std::string s_col("VARIABLE_PAR_INDEX");
  return s_col;
}

std::string ora::PoolMappingElementTable::variableTypeColumn(){
  static std::string s_col("VARIABLE_TYPE");
  return s_col;
}

std::string ora::PoolMappingElementTable::tableNameColumn(){
  static std::string s_col("TABLE_NAME");
  return s_col;  
}

std::string ora::PoolMappingElementTable::columnNameColumn(){
  static std::string s_col("COLUMN_NAME");
  return s_col;  
}

ora::PoolMappingElementTable::PoolMappingElementTable( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::PoolMappingElementTable::~PoolMappingElementTable(){
}

bool ora::PoolMappingElementTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::PoolMappingElementTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "POOL database mapping element table already exists in this schema.",
                    "PoolMappingElementTable::create");
  }
  throwException( "POOL database cannot be created.","PoolMappingElementTable::create");  
}

void ora::PoolMappingElementTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

ora::PoolDbCacheData::PoolDbCacheData():
  m_id( 0 ),
  m_name("" ),
  m_className("" ),
  m_mappingVersion( "" ),
  m_nobjWr( 0 ){
}

ora::PoolDbCacheData::PoolDbCacheData( int id,
                                       const std::string& name,
                                       const std::string& className,
                                       const std::string& mappingVersion,
                                       unsigned int nobjWr ):
  m_id( id ),
  m_name( name ),
  m_className( className ),
  m_mappingVersion( mappingVersion ),
  m_nobjWr( nobjWr ){
}

ora::PoolDbCacheData::~PoolDbCacheData(){
}

ora::PoolDbCacheData::PoolDbCacheData( const ora::PoolDbCacheData& rhs ):
  m_id( rhs.m_id ),
  m_name( rhs.m_name ),
  m_className( rhs.m_className ),
  m_mappingVersion( rhs.m_mappingVersion ),
  m_nobjWr( rhs.m_nobjWr ){
}

ora::PoolDbCacheData& ora::PoolDbCacheData::operator=( const ora::PoolDbCacheData& rhs ){
  m_id = rhs.m_id;
  m_name = rhs.m_name;
  m_className = rhs.m_className;
  m_mappingVersion = rhs.m_mappingVersion;
  m_nobjWr = rhs.m_nobjWr;
  return *this;
}

ora::PoolDbCache::PoolDbCache():
  m_databaseData(),
  m_mappingData(),
  m_idMap(),
  m_sequences(){
  m_databaseData.m_nobjWr = 1;
  m_mappingData.m_nobjWr = 1;
}

ora::PoolDbCache::~PoolDbCache(){
}

void ora::PoolDbCache::add( int id, const PoolDbCacheData& data ){
  std::map<int,PoolDbCacheData >::iterator iData = m_idMap.insert( std::make_pair( id, data )).first;
  std::map<std::string,PoolDbCacheData*>::iterator iS = m_sequences.find( MappingRules::sequenceNameForContainerId() );
  if( iS == m_sequences.end() ){
    throwException( "ContainerId Sequence is empty","PoolDbCache::add");
  }
  if( id > (int)iS->second->m_nobjWr ){
    iS->second->m_nobjWr = id;
  }
  m_sequences.insert( std::make_pair( MappingRules::sequenceNameForContainer( data.m_name ),&iData->second ) );
}

const std::string& ora::PoolDbCache::nameById( int id ){
  PoolDbCacheData& data = find( id );
  return data.m_name;
}

ora::PoolDbCacheData& ora::PoolDbCache::find( int id ){
  std::map<int,PoolDbCacheData >::iterator iC = m_idMap.find( id );
  if( iC == m_idMap.end() ){
    throwException("Container has not been found in the cache.","PoolDbCache::find");    
  }
  return iC->second;
}

void ora::PoolDbCache::remove( int id ){
  std::string name = find( id ).m_name;
  m_sequences.erase( MappingRules::sequenceNameForContainer( name ) );
  m_idMap.erase( id );
}

std::map<std::string,ora::PoolDbCacheData*>& ora::PoolDbCache::sequences(){
  return m_sequences;
}

void ora::PoolDbCache::clear(){
  m_sequences.clear();
  m_idMap.clear();
  m_sequences.insert(std::make_pair( MappingRules::sequenceNameForContainerId(), &m_databaseData ) );
  m_sequences.insert(std::make_pair( MappingRules::sequenceNameForMapping(), &m_mappingData ) );
}

std::string ora::PoolContainerHeaderTable::tableName(){
  static std::string s_name("POOL_RSS_CONTAINERS");
  return s_name;
}


std::string ora::PoolContainerHeaderTable::containerIdColumn(){
  static std::string s_column("CONTAINER_ID");
  return s_column;
}


std::string ora::PoolContainerHeaderTable::containerNameColumn(){
  static std::string s_column("CONTAINER_NAME");
  return s_column;  
}

std::string ora::PoolContainerHeaderTable::containerTypeColumn(){
  static std::string s_column("CONTAINER_TYPE");
  return s_column;  
}

std::string ora::PoolContainerHeaderTable::tableNameColumn(){
  static std::string s_column("TABLE_NAME");
  return s_column;
}

std::string ora::PoolContainerHeaderTable::classNameColumn(){
  static std::string s_column("CLASS_NAME");
  return s_column;
}

std::string ora::PoolContainerHeaderTable::baseMappingVersionColumn(){
  static std::string s_column("MAPPING_VERSION");
  return s_column;
}

std::string ora::PoolContainerHeaderTable::numberOfWrittenObjectsColumn(){
  static std::string s_column("NUMBER_OF_WRITTEN_OBJECTS");
  return s_column;  
}

std::string ora::PoolContainerHeaderTable::numberOfDeletedObjectsColumn(){
  static std::string s_column("NUMBER_OF_DELETED_OBJECTS");
  return s_column;  
}

std::string ora::PoolContainerHeaderTable::homogeneousContainerType(){
  static std::string s_type("Homogeneous");
  return s_type;
}

ora::PoolContainerHeaderTable::PoolContainerHeaderTable( coral::ISchema& dbSchema ):
  m_schema(dbSchema),
  m_dbCache( 0 ){
}

ora::PoolContainerHeaderTable::~PoolContainerHeaderTable(){
}

void ora::PoolContainerHeaderTable::init( PoolDbCache& dbCache ){
  m_dbCache = &dbCache;
}

bool ora::PoolContainerHeaderTable::getContainerData( std::map<std::string,
                                                      ora::ContainerHeaderData>& dest ){
  if(!m_dbCache){
    throwException("Container Table handle has not been initialized.","PoolContainerHeaderTable::getContainerData");
  }
  bool ret = false;
  m_dbCache->clear();
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query( containerTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<int>( containerIdColumn() );
  outputBuffer.extend<std::string>( containerNameColumn() );
  outputBuffer.extend<std::string>( classNameColumn() );
  outputBuffer.extend<std::string>( baseMappingVersionColumn() );
  outputBuffer.extend<unsigned int>( numberOfWrittenObjectsColumn() );
  outputBuffer.extend<unsigned int>( numberOfDeletedObjectsColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( containerIdColumn()  );
  query->addToOutputList( containerNameColumn()  );
  query->addToOutputList( classNameColumn()  );
  query->addToOutputList( baseMappingVersionColumn()  );
  query->addToOutputList( numberOfWrittenObjectsColumn()  );
  query->addToOutputList( numberOfDeletedObjectsColumn() );
  std::stringstream condition;
  condition << containerTypeColumn()<<" = :"<<containerTypeColumn();
  coral::AttributeList condData;
  condData.extend<std::string>( containerTypeColumn() );
  condData[ containerTypeColumn() ].data<std::string>()=homogeneousContainerType();
  query->setCondition( condition.str(), condData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    int containerId = row[ containerIdColumn() ].data< int >() - 1; //POOL starts counting from 1!
    std::string containerName = row[ containerNameColumn()].data< std::string >();
    std::string className = row[ classNameColumn()].data< std::string >();
    std::string baseMappingVersion = row[ baseMappingVersionColumn()].data< std::string >();
    unsigned int numberOfWrittenObjects = row[ numberOfWrittenObjectsColumn()].data< unsigned int >();
    unsigned int numberOfDeletedObjects = row[ numberOfDeletedObjectsColumn()].data< unsigned int >();
    // containers non-homogeneous are ignored.
    dest.insert( std::make_pair( containerName, ContainerHeaderData( containerId, className,
                                                                     numberOfWrittenObjects-numberOfDeletedObjects ) )) ;
    m_dbCache->add( containerId, PoolDbCacheData(containerId, containerName, className, baseMappingVersion, numberOfWrittenObjects) );
  }
  return ret;
}

void ora::PoolContainerHeaderTable::addContainer( int containerId,
                                                  const std::string& containerName,
                                                  const std::string& className ){
  /**
  if(!m_dbCache){
    throwException("Container Table handle has not been initialized.","PoolContainerHeaderTable::addContainer");
  }
  PoolDbCacheData& contData = m_dbCache->find( containerId );
  
  unsigned int nobj = 0;
  coral::AttributeList dataToInsert;
  dataToInsert.extend<int>( containerIdColumn());
  dataToInsert.extend<std::string>( containerNameColumn());
  dataToInsert.extend<std::string>( classNameColumn());
  dataToInsert.extend<std::string>( tableNameColumn());
  dataToInsert.extend<std::string>( baseMappingVersionColumn());
  dataToInsert.extend<unsigned int>( numberOfWrittenObjectsColumn());
  dataToInsert.extend<unsigned int>( numberOfDeletedObjectsColumn());
  dataToInsert[ containerIdColumn() ].data<int>() = containerId;
  dataToInsert[ containerNameColumn() ].data<std::string>() = containerName;
  dataToInsert[ classNameColumn() ].data<std::string>() = className;
  dataToInsert[ tableNameColumn() ].data<std::string>() = "-";
  dataToInsert[ baseMappingVersionColumn() ].data<std::string>() = contData.m_mappingVersion;
  dataToInsert[ numberOfWrittenObjectsColumn() ].data<unsigned int>() = nobj;
  dataToInsert[ numberOfDeletedObjectsColumn() ].data<unsigned int>() = nobj;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  containerTable.dataEditor().insertRow( dataToInsert );
  **/
  throwException( "Cannot create new Containers into POOL database.","PoolContainerHeaderTable::addContainer");  
}

void ora::PoolContainerHeaderTable::removeContainer( int id ){
  if(!m_dbCache){
    throwException("Container Table handle has not been initialized.","PoolContainerHeaderTable::removeContainer");
  }
  m_dbCache->remove( id );
  std::stringstream whereClause;
  whereClause << containerIdColumn() << "= :" <<containerIdColumn();
  coral::AttributeList whereData;
  whereData.extend< int >( containerIdColumn() );
  whereData.begin()->data< int >() = id + 1; //POOL starts counting from 1!;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  containerTable.dataEditor().deleteRows(whereClause.str(),whereData);
}

void ora::PoolContainerHeaderTable::incrementNumberOfObjects( int containerId  ){
  throwException( "Operation not supported into POOL database.","PoolContainerHeaderTable::incrementNumberOfObjects");  
}

void ora::PoolContainerHeaderTable::decrementNumberOfObjects( int containerId  ){
  throwException( "Operation not supported into POOL database.","PoolContainerHeaderTable::decrementNumberOfObjects");  
}

void ora::PoolContainerHeaderTable::updateNumberOfObjects( const std::map<int,unsigned int>& numberOfObjectsForContainerIds ){
  if( numberOfObjectsForContainerIds.size() ){

    if(!m_dbCache){
      throwException("Container Table handle has not been initialized.","PoolContainerHeaderTable::updateNumberOfObjects");
    }

    std::stringstream whereClause;
    whereClause << containerIdColumn() << " = :" <<containerIdColumn();
    std::stringstream setClause;
    setClause << numberOfWrittenObjectsColumn()<< " = :"<<numberOfWrittenObjectsColumn();
    setClause << " , "<< numberOfDeletedObjectsColumn()<< " = :"<<numberOfDeletedObjectsColumn();
    coral::AttributeList updateData;
    updateData.extend<unsigned int>( numberOfWrittenObjectsColumn()  );
    updateData.extend<unsigned int>( numberOfDeletedObjectsColumn()  );
    updateData.extend<int>( containerIdColumn() );

    coral::ITable& containerTable = m_schema.tableHandle( tableName() );
    std::auto_ptr<coral::IBulkOperation> bulkUpdate( containerTable.dataEditor().bulkUpdateRows( setClause.str(), whereClause.str(), updateData,(int)numberOfObjectsForContainerIds.size()+1));

    for( std::map<int,unsigned int>::const_iterator iCont = numberOfObjectsForContainerIds.begin();
         iCont != numberOfObjectsForContainerIds.end(); ++iCont ){

      PoolDbCacheData& contData = m_dbCache->find( iCont->first );
      unsigned int nwrt = contData.m_nobjWr;
      unsigned int ndel = nwrt-iCont->second;

      updateData[containerIdColumn()].data<int>() = iCont->first + 1; //POOL starts counting from 1!;
      updateData[numberOfWrittenObjectsColumn()].data<unsigned int>() = nwrt;
      updateData[numberOfDeletedObjectsColumn()].data<unsigned int>() = ndel;
      bulkUpdate->processNextIteration();

    }
    bulkUpdate->flush();
  }
}

bool ora::PoolContainerHeaderTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::PoolContainerHeaderTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "POOL database container header table already exists in this schema.",
                    "PoolContainerHeaderTable::create");
  }
  throwException( "POOL database cannot be created.","PoolContainerHeaderTable::create");  
}

void ora::PoolContainerHeaderTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::PoolClassVersionTable::tableName(){
  static std::string s_table("POOL_OR_CLASS_VERSIONS");
  return s_table;
}

std::string ora::PoolClassVersionTable::classVersionColumn(){
  static std::string s_col("CLASS_VERSION");
  return s_col;
}

std::string ora::PoolClassVersionTable::containerNameColumn(){
  static std::string s_col("CONTAINER_ID");
  return s_col;

}
 
std::string ora::PoolClassVersionTable::mappingVersionColumn(){
  static std::string s_col("MAPPING_VERSION");
  return s_col;
}

ora::PoolClassVersionTable::PoolClassVersionTable( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::PoolClassVersionTable::~PoolClassVersionTable(){
}

bool ora::PoolClassVersionTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::PoolClassVersionTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "POOL database class version table already exists in this schema.",
                    "PoolClassVersionTable::create");
  }
  throwException( "POOL database cannot be created.","PoolClassVersionTable::create");  
}

void ora::PoolClassVersionTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

namespace ora {
  std::string mappingTypeFromPool( const std::string& mappingType ){
    if( mappingType == "PoolArray" ) return MappingElement::OraArrayMappingElementType();
    return mappingType;
  }

  std::string variableNameFromPool( const std::string& variableName ){
    size_t ind = variableName.find("pool::PVector");
    if( ind != std::string::npos ){
      return "ora::PVector"+variableName.substr(13);
    }
    return variableName;
  }
}

std::string ora::PoolMappingSchema::emptyScope(){
  static std::string s_scope(" ");
  return s_scope;
}

ora::PoolMappingSchema::PoolMappingSchema( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ),
  m_dbCache( 0 ){
}

ora::PoolMappingSchema::~PoolMappingSchema(){
}

void ora::PoolMappingSchema::init( PoolDbCache& dbCache ){
  m_dbCache = &dbCache;
}

bool ora::PoolMappingSchema::getVersionList( std::set<std::string>& dest ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( PoolMappingVersionTable::tableName() ).newQuery() );
  query->addToOutputList( PoolMappingVersionTable::mappingVersionColumn() );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string mappingVersion = currentRow[ PoolMappingVersionTable::mappingVersionColumn()].data<std::string>();
    dest.insert( mappingVersion );
  }
  return ret;
}

namespace ora {
  
  void rebuildPoolMapping( const std::string& scope, const std::string& extraScope, const std::map<std::string, std::vector<MappingRawElement> >& elementsByScope, ora::MappingRawData& dest, int& counter ){
    std::map<std::string, std::vector<MappingRawElement> >::const_iterator iSc = elementsByScope.find( scope );
    if( iSc != elementsByScope.end() ){
      for( std::vector<MappingRawElement>::const_iterator iMap = iSc->second.begin();
           iMap != iSc->second.end(); ++iMap ){
        MappingRawElement& elem = dest.addElement( counter ) = *iMap;
        elem.scopeName = extraScope+"::"+iMap->scopeName;
        counter++;
        rebuildPoolMapping( scope+"::"+iMap->variableName, extraScope, elementsByScope, dest, counter );
      }
    }
  }
}

bool ora::PoolMappingSchema::getMapping( const std::string& version,
                                         ora::MappingRawData& dest ){
  bool ret = false;
  coral::ITable& mappingTable = m_schema.tableHandle( PoolMappingElementTable::tableName() );
  std::auto_ptr<coral::IQuery> query(mappingTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<std::string>( PoolMappingElementTable::elementTypeColumn() );
  outputBuffer.extend<std::string>( PoolMappingElementTable::scopeNameColumn() );
  outputBuffer.extend<std::string>( PoolMappingElementTable::variableNameColumn() );
  outputBuffer.extend<std::string>( PoolMappingElementTable::variableTypeColumn() );
  outputBuffer.extend<std::string>( PoolMappingElementTable::tableNameColumn() );
  outputBuffer.extend<std::string>( PoolMappingElementTable::columnNameColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( PoolMappingElementTable::elementTypeColumn() );
  query->addToOutputList( PoolMappingElementTable::scopeNameColumn() );
  query->addToOutputList( PoolMappingElementTable::variableNameColumn() );
  query->addToOutputList( PoolMappingElementTable::variableTypeColumn() );
  query->addToOutputList( PoolMappingElementTable::tableNameColumn() );
  query->addToOutputList( PoolMappingElementTable::columnNameColumn() );
  std::ostringstream condition;
  condition << PoolMappingElementTable::mappingVersionColumn()<<"= :"<< PoolMappingElementTable::mappingVersionColumn();
  coral::AttributeList condData;
  condData.extend<std::string>( PoolMappingElementTable::mappingVersionColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< std::string >() = version;
  query->setCondition( condition.str(), condData );
  query->addToOrderList( PoolMappingElementTable::scopeNameColumn() );
  query->addToOrderList( PoolMappingElementTable::variableNameColumn() );
  // check the order: column order has to be swapped!
  query->addToOrderList( PoolMappingElementTable::variableParIndexColumn() );
  coral::ICursor& cursor = query->execute();
  std::set<std::string> topElements;
  std::map<std::string,MappingRawElement> elementsByVarName;
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string scope = currentRow[ PoolMappingElementTable::scopeNameColumn() ].data<std::string>();
    std::string varName = currentRow[ PoolMappingElementTable::variableNameColumn() ].data<std::string>();
    std::string elemId = scope+"::"+varName;
    std::map<std::string,MappingRawElement>::iterator iE = elementsByVarName.find( elemId );
    if( iE == elementsByVarName.end() ) {
      iE = elementsByVarName.insert( std::make_pair( elemId, MappingRawElement())).first;
      MappingRawElement& elem = iE->second;
      elem.elementType = mappingTypeFromPool( currentRow[ PoolMappingElementTable::elementTypeColumn() ].data<std::string>() );
      elem.scopeName = scope;
      elem.variableName = variableNameFromPool( varName );
      elem.variableType = currentRow[ PoolMappingElementTable::variableTypeColumn() ].data<std::string>();
      elem.tableName = currentRow[ PoolMappingElementTable::tableNameColumn() ].data<std::string>();
      if(elem.scopeName == emptyScope()) {
        if( elem.elementType == MappingElement::objectMappingElementType()  ){
          if( topElements.find( elemId ) == topElements.end() ){
            topElements.insert( elemId );
          }
        }
      }
    }
    iE->second.columns.push_back( currentRow[ PoolMappingElementTable::columnNameColumn() ].data<std::string>() );
  }
  // re-ordering by scope
  std::map<std::string, std::vector<MappingRawElement> > elementsByScope;
  for( std::map<std::string,MappingRawElement>::iterator iEl = elementsByVarName.begin();
       iEl != elementsByVarName.end(); ++iEl ){
    // reversing the columns
    std::vector<std::string> reverseCols;
    for( std::vector<std::string>::reverse_iterator iR = iEl->second.columns.rbegin();
         iR != iEl->second.columns.rend(); ++iR ){
      reverseCols.push_back( *iR );
    }
    iEl->second.columns = reverseCols;
    std::string scope = iEl->second.scopeName;
    if( scope != emptyScope() ){
      std::map<std::string, std::vector<MappingRawElement> >::iterator iS = elementsByScope.find( scope );
      if( iS == elementsByScope.end() ){
        elementsByScope.insert( std::make_pair( scope, std::vector<MappingRawElement>(1,iEl->second ) ));
      } else {
        iS->second.push_back( iEl->second );
      }
    }
  }
  // rebuilding + adding class elements  
  int eid = 0;
  for( std::set<std::string>::const_iterator iEl = topElements.begin();
       iEl != topElements.end(); ++iEl ){
    // adding the class elements...
    std::map<std::string,MappingRawElement>::iterator iE = elementsByVarName.find( *iEl );
    MappingRawElement classElement = iE->second;
    classElement.elementType = MappingElement::classMappingElementType();
    classElement.scopeName = MappingRawElement::emptyScope();
    dest.addElement( eid ) = classElement;
    eid++;
    MappingRawElement firstElement = iE->second;
    firstElement.scopeName = iE->second.variableName;
    dest.addElement( eid ) = firstElement;
    eid++;
    // rebuilding extending the scope...
    rebuildPoolMapping( iE->second.variableName, iE->second.variableName, elementsByScope, dest, eid );
  }       
  return ret;
}

void ora::PoolMappingSchema::storeMapping( const MappingRawData& mapping ){
  // first update the version table
  coral::ITable& mappingVersionTable = m_schema.tableHandle( PoolMappingVersionTable::tableName() );
  coral::AttributeList  rowBuffer;
  rowBuffer.extend< std::string >( PoolMappingVersionTable::mappingVersionColumn() );
  rowBuffer[ PoolMappingVersionTable::mappingVersionColumn() ].data<std::string>()= mapping.version;
  mappingVersionTable.dataEditor().insertRow( rowBuffer );

  // then update the element tables
  coral::ITable& mappingElementTable = m_schema.tableHandle( PoolMappingElementTable::tableName() );
  coral::AttributeList  dataBuffer;
  dataBuffer.extend< std::string >( PoolMappingElementTable::mappingVersionColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::elementIdColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::elementTypeColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::scopeNameColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::variableNameColumn() );
  dataBuffer.extend< unsigned int >( PoolMappingElementTable::variableParIndexColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::variableTypeColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::tableNameColumn() );
  dataBuffer.extend< std::string >( PoolMappingElementTable::columnNameColumn() );
  dataBuffer[ PoolMappingElementTable::mappingVersionColumn() ].data<std::string>()= mapping.version;

  for( std::map < int, MappingRawElement >::const_iterator iElem = mapping.elements.begin();
       iElem != mapping.elements.end(); iElem++ ){
    for( size_t iParamIndex = 0; iParamIndex < iElem->second.columns.size(); iParamIndex++ ){
      std::stringstream elemIdx;
      elemIdx << iElem->first;
      std::string scopeName = iElem->second.scopeName;
      if( scopeName == MappingRawElement::emptyScope() ) scopeName = std::string(" ");
      dataBuffer[ PoolMappingElementTable::elementIdColumn() ].data<std::string>() = elemIdx.str();
      dataBuffer[ PoolMappingElementTable::elementTypeColumn()].data<std::string>()=  iElem->second.elementType;
      dataBuffer[ PoolMappingElementTable::scopeNameColumn() ].data<std::string>()= scopeName;
      dataBuffer[ PoolMappingElementTable::variableNameColumn() ].data<std::string>()= iElem->second.variableName;
      dataBuffer[ PoolMappingElementTable::variableParIndexColumn() ].data<unsigned int>() = iParamIndex;
      dataBuffer[ PoolMappingElementTable::variableTypeColumn() ].data<std::string>()= iElem->second.variableType;
      dataBuffer[ PoolMappingElementTable::tableNameColumn() ].data<std::string>()= iElem->second.tableName;
      dataBuffer[ PoolMappingElementTable::columnNameColumn() ].data<std::string>()= iElem->second.columns[iParamIndex];
      mappingElementTable.dataEditor().insertRow( dataBuffer );
    }
  }
}

void ora::PoolMappingSchema::removeMapping( const std::string& version ){
  // Remove all rows in the tables with the version.
  coral::AttributeList whereData;
  whereData.extend<std::string>( PoolMappingVersionTable::mappingVersionColumn() );
  whereData.begin()->data<std::string>() = version;

  std::string condition = PoolMappingVersionTable::mappingVersionColumn() + " = :" + PoolMappingVersionTable::mappingVersionColumn();
  m_schema.tableHandle( PoolClassVersionTable::tableName() ).dataEditor().deleteRows( condition, whereData );
  m_schema.tableHandle( PoolMappingElementTable::tableName() ).dataEditor().deleteRows( condition, whereData );
  m_schema.tableHandle( PoolMappingVersionTable::tableName() ).dataEditor().deleteRows( condition, whereData );
}

bool ora::PoolMappingSchema::getContainerTableMap( std::map<std::string, int>&){
  // not implemented for the moment
  return false;
}

bool ora::PoolMappingSchema::getMappingVersionListForContainer( int containerId,
                                                                std::set<std::string>& dest,
                                                                bool onlyDependency ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.newQuery() );
  query->addToTableList( PoolClassVersionTable::tableName(), "T0" );
  query->addToTableList( PoolContainerHeaderTable::tableName(), "T1" );
  query->addToTableList( PoolMappingElementTable::tableName(), "T2" );
  query->addToOutputList( "T0."+PoolClassVersionTable::mappingVersionColumn() );
  query->setDistinct();
  std::ostringstream condition;
  condition <<"T0."<<PoolClassVersionTable::containerNameColumn()<<" = "<<"T1."<<PoolContainerHeaderTable::containerNameColumn();
  condition <<" AND T0."<<PoolClassVersionTable::mappingVersionColumn()<<" = "<<"T2."<<PoolMappingElementTable::mappingVersionColumn();
  condition <<" AND T1."<<PoolContainerHeaderTable::containerIdColumn()<<" =:"<<PoolContainerHeaderTable::containerIdColumn();
  coral::AttributeList condData;
  condData.extend< int >( PoolContainerHeaderTable::containerIdColumn() );
  condData[ PoolContainerHeaderTable::containerIdColumn() ].data< int >() = containerId + 1; //POOL starts counting from 1!;
  if( onlyDependency ){
    condition <<" AND T2."<<PoolMappingElementTable::elementTypeColumn() <<" = :"<<PoolMappingElementTable::elementTypeColumn();
    condData.extend< std::string >( PoolMappingElementTable::elementTypeColumn() );
    condData[PoolMappingElementTable::elementTypeColumn() ].data<std::string>() = MappingElement::dependencyMappingElementType();
  }
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string mappingVersion = currentRow[ "T0."+PoolClassVersionTable::mappingVersionColumn() ].data<std::string>();
    dest.insert( mappingVersion );
  }
  return ret;
}


bool ora::PoolMappingSchema::getDependentClassesInContainerMapping( int,
                                                                    std::set<std::string>& ){
  // not implemented for the moment
  return false;
}

bool ora::PoolMappingSchema::getClassVersionListForMappingVersion( const std::string& mappingVersion,
                                                                   std::set<std::string>& destination ){
  
  bool ret = false;
  coral::ITable& classVersionTable = m_schema.tableHandle( PoolClassVersionTable::tableName() );
  std::auto_ptr<coral::IQuery> query( classVersionTable.newQuery() );
  query->setDistinct();
  query->addToOutputList( PoolClassVersionTable::classVersionColumn() );
  std::ostringstream condition;
  condition <<PoolClassVersionTable::mappingVersionColumn()<<" =:"<<PoolClassVersionTable::mappingVersionColumn();
  coral::AttributeList condData;
  condData.extend<std::string>(PoolClassVersionTable::mappingVersionColumn());
  condData[ PoolClassVersionTable::mappingVersionColumn() ].data< std::string >() = mappingVersion;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string classVersion = currentRow[ PoolClassVersionTable::classVersionColumn() ].data<std::string>();
    destination.insert( classVersion );
  }
  return ret;
}

bool ora::PoolMappingSchema::getClassVersionListForContainer( int containerId,
                                                              std::map<std::string,std::string>& versionMap ){
  
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.newQuery() );
  query->addToTableList( PoolClassVersionTable::tableName(), "T0" );
  query->addToTableList( PoolContainerHeaderTable::tableName(), "T1" );
  query->addToOutputList( "T0."+PoolClassVersionTable::classVersionColumn() );
  query->addToOutputList( "T0."+PoolClassVersionTable::mappingVersionColumn() );
  query->setDistinct();
  std::ostringstream condition;
  condition <<"T0."<<PoolClassVersionTable::containerNameColumn()<<" = "<<"T1."<<PoolContainerHeaderTable::containerNameColumn();
  condition <<" AND T1."<<PoolContainerHeaderTable::containerIdColumn()<<" =:"<<PoolContainerHeaderTable::containerIdColumn();
  coral::AttributeList condData;
  condData.extend< int >( PoolContainerHeaderTable::containerIdColumn() );
  condData[ PoolContainerHeaderTable::containerIdColumn() ].data< int >() = containerId + 1; //POOL starts counting from 1!;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string classVersion = currentRow[ "T0."+PoolClassVersionTable::classVersionColumn() ].data<std::string>();
    std::string mappingVersion = currentRow[ "T0."+PoolClassVersionTable::mappingVersionColumn() ].data<std::string>();
    versionMap.insert( std::make_pair(classVersion, mappingVersion ) );
  }
  return ret;
}

bool ora::PoolMappingSchema::getMappingVersionListForTable( const std::string&,
                                                            std::set<std::string>& ){
  // not implemented for the moment
  return false;
}

bool ora::PoolMappingSchema::selectMappingVersion( const std::string& classId,
                                                   int containerId,
                                                   std::string& destination ){
  bool ret = false;
  destination.clear();

  std::pair<bool,std::string> isBaseId = MappingRules::classNameFromBaseId( classId );
  if( !isBaseId.first ){
    std::auto_ptr<coral::IQuery> query( m_schema.newQuery() );
    query->addToTableList( PoolClassVersionTable::tableName(), "T0" );
    query->addToTableList( PoolContainerHeaderTable::tableName(), "T1" );
    query->addToOutputList( "T0."+PoolClassVersionTable::mappingVersionColumn() );
    std::ostringstream condition;
    condition <<"T0."<<PoolClassVersionTable::containerNameColumn()<<" = "<<"T1."<<PoolContainerHeaderTable::containerNameColumn();
    condition << " AND T0."<<PoolClassVersionTable::classVersionColumn() << " =:" <<PoolClassVersionTable::classVersionColumn();
    condition << " AND T1."<<PoolContainerHeaderTable::containerIdColumn() << " =:" <<PoolContainerHeaderTable::containerIdColumn();
    coral::AttributeList condData;
    condData.extend<std::string>( PoolClassVersionTable::classVersionColumn() );
    condData.extend<int>( PoolContainerHeaderTable::containerIdColumn() );
    coral::AttributeList::iterator iAttribute = condData.begin();
    iAttribute->data< std::string >() = MappingRules::classVersionFromId( classId );
    ++iAttribute;
    iAttribute->data< int >() = containerId + 1; //POOL starts counting from 1!;
    query->setCondition( condition.str(), condData );
    coral::ICursor& cursor = query->execute();
    while ( cursor.next() ) {
      ret = true;
      const coral::AttributeList& currentRow = cursor.currentRow();
      destination = currentRow["T0."+PoolClassVersionTable::mappingVersionColumn()].data<std::string>();
    }
  } else {
    PoolDbCacheData& containerData = m_dbCache->find( containerId );
    // in POOL db this will be only possible for top level classes (not for dependencies)
    if( containerData.m_className == isBaseId.second ){
      destination = containerData.m_mappingVersion;
      ret = true;
    }
  }
  
  return ret;  
}

bool ora::PoolMappingSchema::containerForMappingVersion( const std::string&,
                                                         int& ){
  // not implemented for the moment
  return false;
}

void ora::PoolMappingSchema::insertClassVersion( const std::string&, //className
                                                 const std::string& classVersion,
                                                 const std::string&, //classId
                                                 int, // dependencyIndex,
                                                 int containerId,
                                                 const std::string& mappingVersion ){
  if(!m_dbCache){
    throwException("MappingSchema handle has not been initialized.","PoolMappingSchema::insertClassVersion");
  }
  
  coral::ITable& classVersionTable = m_schema.tableHandle( PoolClassVersionTable::tableName() );
  coral::AttributeList inputData;
  inputData.extend<std::string>( PoolClassVersionTable::mappingVersionColumn());
  inputData.extend<std::string>( PoolClassVersionTable::classVersionColumn());
  inputData.extend<std::string>( PoolClassVersionTable::containerNameColumn());
  
  std::string containerName = m_dbCache->nameById( containerId );
  coral::AttributeList::iterator iInAttr = inputData.begin();
  iInAttr->data< std::string >() = mappingVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = classVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = containerName;
  classVersionTable.dataEditor().insertRow( inputData );
}

void ora::PoolMappingSchema::setMappingVersion( const std::string& classId,
                                                int containerId,
                                                const std::string& mappingVersion ){
  if(!m_dbCache){
    throwException("MappingSchema handle has not been initialized.","PoolMappingSchema::setMappingVersion");
  }
  coral::ITable& classVersionTable = m_schema.tableHandle( PoolClassVersionTable::tableName() );
  coral::AttributeList inputData;
  inputData.extend<std::string>( PoolClassVersionTable::mappingVersionColumn());
  inputData.extend<std::string>( PoolClassVersionTable::classVersionColumn());
  inputData.extend<std::string>( PoolClassVersionTable::containerNameColumn());
  std::string classVersion = MappingRules::classVersionFromId( classId );
  std::string containerName = m_dbCache->nameById( containerId );
  coral::AttributeList::iterator iInAttr = inputData.begin();
  iInAttr->data< std::string >() = mappingVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = classVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = containerName;
  std::string setClause = PoolClassVersionTable::mappingVersionColumn()+" =:"+ PoolClassVersionTable::mappingVersionColumn();
  std::string whereClause = PoolClassVersionTable::classVersionColumn()+" =:"+ PoolClassVersionTable::classVersionColumn()+" AND "+
    PoolClassVersionTable::containerNameColumn()+" =:"+ PoolClassVersionTable::containerNameColumn();
  classVersionTable.dataEditor().updateRows( setClause,whereClause, inputData  );
}

bool ora::PoolDatabaseSchema::existsMainTable( coral::ISchema& dbSchema ){
  PoolMainTable tmp( dbSchema );
  return tmp.exists();
}

ora::PoolDatabaseSchema::PoolDatabaseSchema( coral::ISchema& dbSchema ):
  IDatabaseSchema( dbSchema ),
  m_schema( dbSchema ),
  m_dbCache(),
  m_mainTable( dbSchema ),
  m_sequenceTable( dbSchema ),
  m_mappingVersionTable( dbSchema ),
  m_mappingElementTable( dbSchema ),
  m_containerHeaderTable( dbSchema ),
  m_classVersionTable( dbSchema ),
  m_mappingSchema( dbSchema ){
  m_sequenceTable.init( m_dbCache );
  m_containerHeaderTable.init( m_dbCache );
  m_mappingSchema.init( m_dbCache );
}

ora::PoolDatabaseSchema::~PoolDatabaseSchema(){
}

bool ora::PoolDatabaseSchema::exists(){
  if(!m_mainTable.exists()){
    return false;
  }
  if(!m_sequenceTable.exists() ||
     !m_mappingVersionTable.exists() ||
     !m_mappingElementTable.exists() ||
     !m_containerHeaderTable.exists() ||
     !m_classVersionTable.exists()){
    throwException( "POOL database is corrupted..",
                    "PoolDatabaseSchema::exists");
  }
  return true;
}

void ora::PoolDatabaseSchema::create(){
  throwException( "POOL database cannot be created.","PoolDatabaseSchema::create");  
}

void ora::PoolDatabaseSchema::drop(){
  m_classVersionTable.drop();
  m_mappingElementTable.drop();
  m_sequenceTable.drop();
  m_containerHeaderTable.drop();
  m_mappingVersionTable.drop();
  m_mainTable.drop(); 
}

ora::IMainTable& ora::PoolDatabaseSchema::mainTable(){
  return m_mainTable;
}

ora::ISequenceTable& ora::PoolDatabaseSchema::sequenceTable(){
  return m_sequenceTable;
}

ora::IDatabaseTable& ora::PoolDatabaseSchema::mappingVersionTable(){
  return m_mappingVersionTable;  
}

ora::IDatabaseTable& ora::PoolDatabaseSchema::mappingElementTable(){
  return m_mappingElementTable;  
}

ora::IContainerHeaderTable& ora::PoolDatabaseSchema::containerHeaderTable(){
  return m_containerHeaderTable;
}

ora::IDatabaseTable& ora::PoolDatabaseSchema::classVersionTable(){
  return m_classVersionTable;  
}

ora::IMappingSchema& ora::PoolDatabaseSchema::mappingSchema(){
  return m_mappingSchema;  
}

  
    
