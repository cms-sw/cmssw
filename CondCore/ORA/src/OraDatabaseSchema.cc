#include "CondCore/ORA/interface/Exception.h"
#include "OraDatabaseSchema.h"
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

std::string ora::OraMainTable::version(){
  static std::string s_version("1.1.0");
  return s_version;
}

std::string ora::OraMainTable::tableName(){
  static std::string s_name("ORA_DB");
  return s_name;
}

std::string ora::OraMainTable::parameterNameColumn(){
  static std::string s_column("PARAMETER_NAME");
  return s_column;
}

std::string ora::OraMainTable::parameterValueColumn(){
  static std::string s_column("PARAMETER_VALUE");
  return s_column;
}

ora::OraMainTable::OraMainTable( coral::ISchema& dbSchema ):
  m_schema( dbSchema ){
}

ora::OraMainTable::~OraMainTable(){
}

void ora::OraMainTable::setParameter( const std::string& paramName, 
                                      const std::string& paramValue ){
  if( !paramName.empty() && !paramValue.empty() ){
    coral::ITable& table = m_schema.tableHandle( tableName() );
    coral::AttributeList dataToInsert;
    dataToInsert.extend<std::string>( parameterNameColumn());
    dataToInsert.extend<std::string>( parameterValueColumn());
    dataToInsert[ parameterNameColumn() ].data<std::string>() = paramName;
    dataToInsert[ parameterValueColumn() ].data<std::string>() = paramValue;
    table.dataEditor().insertRow( dataToInsert );
  }
}

bool ora::OraMainTable::getParameters( std::map<std::string,std::string>& dest ){
  bool ret = false;
  coral::ITable& mainTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query(mainTable.newQuery());
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    std::string paramName = row[ parameterNameColumn()].data< std::string >();
    std::string paramValue = row[ parameterValueColumn()].data< std::string >();
    dest.insert( std::make_pair( paramName, paramValue ) );
  }
  return ret;
}

std::string ora::OraMainTable::schemaVersion(){
  // could be replaced by a call to getParameters in case of needs to distinguish between ora db schema versions...
  return version();
}

bool ora::OraMainTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraMainTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA database main table already exists in this schema.",
                    "OraMainTable::create");
  }
  
  coral::TableDescription descr( "OraDb" );
  descr.setName( tableName() );
  descr.insertColumn( parameterNameColumn(),
                      coral::AttributeSpecification::typeNameForType<std::string>() );
  descr.insertColumn( parameterValueColumn(),
                      coral::AttributeSpecification::typeNameForType<std::string>() );
  descr.setNotNullConstraint( parameterNameColumn() );
  descr.setNotNullConstraint( parameterValueColumn() );
  descr.setPrimaryKey( std::vector<std::string>( 1, parameterNameColumn() ) );

  coral::ITable& table = m_schema.createTable( descr );
  table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );

  coral::AttributeList dataToInsert;
  dataToInsert.extend<std::string>( parameterNameColumn());
  dataToInsert.extend<std::string>( parameterValueColumn());
  dataToInsert[ parameterNameColumn() ].data<std::string>() = IMainTable::versionParameterName();
  dataToInsert[ parameterValueColumn() ].data<std::string>() = version();
  table.dataEditor().insertRow( dataToInsert );
}

void ora::OraMainTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::OraSequenceTable::tableName(){
  static std::string s_name("ORA_SEQUENCE");
  return s_name;
}

std::string ora::OraSequenceTable::sequenceNameColumn(){
  static std::string s_column("NAME");
  return s_column;
}

std::string ora::OraSequenceTable::sequenceValueColumn(){
  static std::string s_column("VALUE");
  return s_column;
}

ora::OraSequenceTable::OraSequenceTable( coral::ISchema& schema ):
  m_schema( schema){
}

ora::OraSequenceTable::~OraSequenceTable(){
}

bool
ora::OraSequenceTable::add( const std::string& sequenceName ){
  // Create the entry in the table 
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
ora::OraSequenceTable::getLastId( const std::string& sequenceName,
                                  int& lastId ) {
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

void ora::OraSequenceTable::sinchronize( const std::string& sequenceName,
                                         int lastValue ){
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

void ora::OraSequenceTable::erase( const std::string& name ){
  coral::AttributeList whereData;
  whereData.extend<std::string>(sequenceNameColumn());
  whereData[ sequenceNameColumn() ].data<std::string>() = name;
  std::string whereClause( sequenceNameColumn() + " = :" + sequenceNameColumn() );
  m_schema.tableHandle( tableName() ).dataEditor().deleteRows( whereClause, whereData );
}

bool ora::OraSequenceTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraSequenceTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA database sequence table already exists in this schema.",
                    "OraSequenceTable::create");
  }
  
  coral::TableDescription description( "OraDb" );
  description.setName( tableName() );

  description.insertColumn( sequenceNameColumn(), coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint( sequenceNameColumn() );

  description.insertColumn( sequenceValueColumn(),coral::AttributeSpecification::typeNameForType<int>() );
  description.setNotNullConstraint( sequenceValueColumn() );

  description.setPrimaryKey( std::vector< std::string >( 1, sequenceNameColumn() ) );
  m_schema.createTable( description ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
}

void ora::OraSequenceTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::OraMappingVersionTable::tableName(){
  static std::string s_table("ORA_MAPPING_VERSION");
  return s_table;
}

std::string ora::OraMappingVersionTable::mappingVersionColumn(){
  static std::string s_col("MAPPING_VERSION");
  return s_col;
}

ora::OraMappingVersionTable::OraMappingVersionTable( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::OraMappingVersionTable::~OraMappingVersionTable(){
}

bool ora::OraMappingVersionTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraMappingVersionTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA database mapping version table already exists in this schema.",
                    "OraMappingVersionTable::create");
  }
  // version table
  coral::TableDescription  description0( "OraDb" );
  description0.setName( tableName() );
  description0.insertColumn( mappingVersionColumn(),
                             coral::AttributeSpecification::typeNameForType<std::string>(), 1000, false);
  description0.setNotNullConstraint( mappingVersionColumn() );
  description0.setPrimaryKey( mappingVersionColumn() );
  m_schema.createTable( description0 ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
}

void ora::OraMappingVersionTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}


std::string ora::OraMappingElementTable::tableName(){
  static std::string s_table("ORA_MAPPING_ELEMENT");
  return s_table;  
}

std::string ora::OraMappingElementTable::mappingVersionColumn(){
  static std::string s_col("MAPPING_VERSION");
  return s_col;  
}

std::string ora::OraMappingElementTable::elementIdColumn(){
  static std::string s_col("ELEMENT_ID");
  return s_col;
}

std::string ora::OraMappingElementTable::elementTypeColumn(){
  static std::string s_col("ELEMENT_TYPE");
  return s_col;  
}

std::string ora::OraMappingElementTable::scopeNameColumn(){
  static std::string s_col("VARIABLE_SCOPE");
  return s_col;
}

std::string ora::OraMappingElementTable::variableNameColumn(){
  static std::string s_col("VARIABLE_NAME");
  return s_col;
}

std::string ora::OraMappingElementTable::variableParIndexColumn(){
  static std::string s_col("VARIABLE_PAR_INDEX");
  return s_col;
}

std::string ora::OraMappingElementTable::variableTypeColumn(){
  static std::string s_col("VARIABLE_TYPE");
  return s_col;

}

std::string ora::OraMappingElementTable::tableNameColumn(){
  static std::string s_col("TABLE_NAME");
  return s_col;  
}

std::string ora::OraMappingElementTable::columnNameColumn(){
  static std::string s_col("COLUMN_NAME");
  return s_col;  
}

ora::OraMappingElementTable::OraMappingElementTable( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::OraMappingElementTable::~OraMappingElementTable(){
}

bool ora::OraMappingElementTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraMappingElementTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA database mapping element table already exists in this schema.",
                    "OraMappingElementTable::create");
  }

  // mapping elements table
  coral::TableDescription description2( "OraDb" );
  description2.setName( tableName() );
  description2.insertColumn(  mappingVersionColumn(),
                              coral::AttributeSpecification::typeNameForType<std::string>(), 1000, false );
  description2.setNotNullConstraint(  mappingVersionColumn() );
  description2.insertColumn(  elementIdColumn(),
                              coral::AttributeSpecification::typeNameForType<int>() );
  description2.setNotNullConstraint(  elementIdColumn() );
  description2.insertColumn(  elementTypeColumn(),
                              coral::AttributeSpecification::typeNameForType<std::string>() );
  description2.setNotNullConstraint(  elementTypeColumn() );
  description2.insertColumn(  scopeNameColumn(),
                              coral::AttributeSpecification::typeNameForType<std::string>() );
  description2.setNotNullConstraint(  scopeNameColumn() );
  description2.insertColumn( variableNameColumn(),
                             coral::AttributeSpecification::typeNameForType<std::string>() );
  description2.setNotNullConstraint( variableNameColumn() );
  description2.insertColumn( variableParIndexColumn(),
                              coral::AttributeSpecification::typeNameForType<unsigned int>() );
  description2.setNotNullConstraint(  variableParIndexColumn() );
  description2.insertColumn(  variableTypeColumn(),
                              coral::AttributeSpecification::typeNameForType<std::string>() );
  description2.setNotNullConstraint(  variableTypeColumn() );
  description2.insertColumn(  tableNameColumn(),
                              coral::AttributeSpecification::typeNameForType<std::string>() );
  description2.setNotNullConstraint(  tableNameColumn() );
  description2.insertColumn(  columnNameColumn(),
                              coral::AttributeSpecification::typeNameForType<std::string>() );
  description2.setNotNullConstraint(  columnNameColumn() );
  std::vector<std::string> cols2;
  cols2.push_back(  elementIdColumn() );
  cols2.push_back(  variableParIndexColumn() );
  description2.setPrimaryKey( cols2 );
  std::string fkName20 = mappingVersionColumn()+"_FK_1";
  description2.createForeignKey( fkName20, mappingVersionColumn(),
                                 OraMappingVersionTable::tableName(),OraMappingVersionTable::mappingVersionColumn());
  m_schema.createTable( description2 ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
}

void ora::OraMappingElementTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::OraContainerHeaderTable::tableName(){
  static std::string s_name("ORA_CONTAINER");
  return s_name;
}


std::string ora::OraContainerHeaderTable::containerIdColumn(){
  static std::string s_column("CONTAINER_ID");
  return s_column;
}


std::string ora::OraContainerHeaderTable::containerNameColumn(){
  static std::string s_column("CONTAINER_NAME");
  return s_column;  
}

std::string ora::OraContainerHeaderTable::classNameColumn(){
  static std::string s_column("CLASS_NAME");
  return s_column;
}

std::string ora::OraContainerHeaderTable::numberOfObjectsColumn(){
  static std::string s_column("NUMBER_OF_OBJECTS");
  return s_column;  
}

ora::OraContainerHeaderTable::OraContainerHeaderTable( coral::ISchema& dbSchema ):
  m_schema(dbSchema){
}

ora::OraContainerHeaderTable::~OraContainerHeaderTable(){
}

bool ora::OraContainerHeaderTable::getContainerData( std::map<std::string,
                                                     ora::ContainerHeaderData>& dest ){
  bool ret = false;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query( containerTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<int>( containerIdColumn() );
  outputBuffer.extend<std::string>( containerNameColumn() );
  outputBuffer.extend<std::string>( classNameColumn() );
  outputBuffer.extend<unsigned int>( numberOfObjectsColumn() );
  query->defineOutput( outputBuffer );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    int containerId = row[ containerIdColumn() ].data< int >();
    std::string containerName = row[ containerNameColumn()].data< std::string >();
    std::string className = row[ classNameColumn()].data< std::string >();
    unsigned int numberOfObjects = row[ numberOfObjectsColumn()].data< unsigned int >();
    dest.insert( std::make_pair( containerName, ContainerHeaderData( containerId, className, numberOfObjects ) )) ;
  }
  return ret;
}

void ora::OraContainerHeaderTable::addContainer( int containerId,
                                                 const std::string& containerName,
                                                 const std::string& className ){
  coral::AttributeList dataToInsert;
  unsigned int numberOfObjects = 0;
  dataToInsert.extend<int>( containerIdColumn());
  dataToInsert.extend<std::string>( containerNameColumn());
  dataToInsert.extend<std::string>( classNameColumn());
  dataToInsert.extend<unsigned int>( numberOfObjectsColumn());
  dataToInsert[ containerIdColumn() ].data<int>() = containerId;
  dataToInsert[ containerNameColumn() ].data<std::string>() = containerName;
  dataToInsert[ classNameColumn() ].data<std::string>() = className;
  dataToInsert[ numberOfObjectsColumn() ].data<unsigned int>() = numberOfObjects;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  containerTable.dataEditor().insertRow( dataToInsert );
}

void ora::OraContainerHeaderTable::removeContainer( int id ){
  std::stringstream whereClause;
  whereClause << containerIdColumn() << "= :" <<containerIdColumn();
  coral::AttributeList whereData;
  whereData.extend< int >( containerIdColumn() );
  whereData.begin()->data< int >() = id;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  containerTable.dataEditor().deleteRows(whereClause.str(),whereData);
}

void ora::OraContainerHeaderTable::updateContainer( int containerId,
                                                    const std::string& setClause ){
  std::stringstream whereClause;
  whereClause << containerIdColumn() << "= :" <<containerIdColumn();
  coral::AttributeList updateData;
  updateData.extend<int>( containerIdColumn() );
  updateData.begin()->data<int>() = containerId;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  containerTable.dataEditor().updateRows(setClause,whereClause.str(),updateData);
}

void ora::OraContainerHeaderTable::incrementNumberOfObjects( int containerId  ){
  std::stringstream setClause;
  setClause << numberOfObjectsColumn() << " = " <<numberOfObjectsColumn() << " + 1";
  return updateContainer( containerId, setClause.str() );
}

void ora::OraContainerHeaderTable::decrementNumberOfObjects( int containerId  ){
  std::stringstream setClause;
  setClause << numberOfObjectsColumn() << " = " <<numberOfObjectsColumn() << " - 1";
  return updateContainer( containerId, setClause.str() );
}

void ora::OraContainerHeaderTable::updateNumberOfObjects( const std::map<int,unsigned int>& numberOfObjectsForContainerIds ){
  if( numberOfObjectsForContainerIds.size() ){

    std::stringstream whereClause;
    whereClause << containerIdColumn() << "= :" <<containerIdColumn();
    std::stringstream setClause;
    setClause << numberOfObjectsColumn() << " = :" <<numberOfObjectsColumn();
    coral::AttributeList updateData;
    updateData.extend<unsigned int>( numberOfObjectsColumn() );
    updateData.extend<int>( containerIdColumn() );

    coral::ITable& containerTable = m_schema.tableHandle( tableName() );
    std::auto_ptr<coral::IBulkOperation> bulkUpdate( containerTable.dataEditor().bulkUpdateRows( setClause.str(), whereClause.str(), updateData,(int)numberOfObjectsForContainerIds.size()));

    for( std::map<int,unsigned int>::const_iterator iCont = numberOfObjectsForContainerIds.begin();
         iCont != numberOfObjectsForContainerIds.end(); ++iCont ){
      updateData[containerIdColumn()].data<int>() = iCont->first;
      updateData[numberOfObjectsColumn()].data<unsigned int>() = iCont->second;
      bulkUpdate->processNextIteration();
    }
    bulkUpdate->flush();
  }
}

bool ora::OraContainerHeaderTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraContainerHeaderTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA database container header table already exists in this schema.",
                    "OraContainerHeaderTable::create");
  }
  
  coral::TableDescription descr( "OraDb" );
  descr.setName( tableName() );
  descr.insertColumn( containerIdColumn(),
                      coral::AttributeSpecification::typeNameForType<int>() );
  descr.insertColumn( containerNameColumn(),
                      coral::AttributeSpecification::typeNameForType<std::string>() );
  descr.insertColumn( classNameColumn(),
                      coral::AttributeSpecification::typeNameForType<std::string>() );
  descr.insertColumn( numberOfObjectsColumn(),
                      coral::AttributeSpecification::typeNameForType<unsigned int>() );
  descr.setNotNullConstraint( containerIdColumn() );
  descr.setNotNullConstraint( containerNameColumn() );
  descr.setNotNullConstraint( classNameColumn() );
  descr.setNotNullConstraint( numberOfObjectsColumn() );
  descr.setPrimaryKey( std::vector<std::string>( 1, containerIdColumn() ) );
  descr.setUniqueConstraint( containerNameColumn() );
  coral::ITable& table = m_schema.createTable( descr );
  table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
}

void ora::OraContainerHeaderTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

std::string ora::OraClassVersionTable::tableName(){
  static std::string s_table("ORA_CLASS_VERSION");
  return s_table;
}

std::string ora::OraClassVersionTable::classNameColumn(){
  static std::string s_col("CLASS_NAME");
  return s_col;
}

std::string ora::OraClassVersionTable::classVersionColumn(){
  static std::string s_col("CLASS_VERSION");
  return s_col;
}

std::string ora::OraClassVersionTable::classIdColumn(){
  static std::string s_col("CLASS_ID");
  return s_col;
}

std::string ora::OraClassVersionTable::dependencyIndexColumn(){
  static std::string s_col("DEPENDENCY_INDEX");
  return s_col;
}

std::string ora::OraClassVersionTable::containerIdColumn(){
  static std::string s_col("CONTAINER_ID");
  return s_col;

}
 
std::string ora::OraClassVersionTable::mappingVersionColumn(){
  static std::string s_col("MAPPING_VERSION");
  return s_col;
}


ora::OraClassVersionTable::OraClassVersionTable( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::OraClassVersionTable::~OraClassVersionTable(){
}

bool ora::OraClassVersionTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraClassVersionTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA database class version table already exists in this schema.",
                    "OraClassVersionTable::create");
  }
  // class version table
  coral::TableDescription description1( "OraDb" );
  description1.setName( tableName() );
  description1.insertColumn( classNameColumn(),
                             coral::AttributeSpecification::typeNameForType<std::string>(), 1000, false);
  description1.setNotNullConstraint( classNameColumn() );
  description1.insertColumn( classVersionColumn(),
                             coral::AttributeSpecification::typeNameForType<std::string>(), 1000, false);
  description1.setNotNullConstraint( classVersionColumn() );
  description1.insertColumn( classIdColumn(),
                             coral::AttributeSpecification::typeNameForType<std::string>(), 1000, false);
  description1.setNotNullConstraint( classIdColumn() );
  description1.insertColumn( dependencyIndexColumn(),
                             coral::AttributeSpecification::typeNameForType<int>() );
  description1.setNotNullConstraint(  dependencyIndexColumn() );
  description1.insertColumn( containerIdColumn(),
                             coral::AttributeSpecification::typeNameForType<int>() );
  description1.setNotNullConstraint(  containerIdColumn() );
  description1.insertColumn( mappingVersionColumn(),
                             coral::AttributeSpecification::typeNameForType<std::string>(), 1000, false);
  description1.setNotNullConstraint( mappingVersionColumn() );
  std::vector<std::string> cols1;
  cols1.push_back( classIdColumn() );
  cols1.push_back( containerIdColumn() );
  description1.setPrimaryKey( cols1 );
  std::string fk10Name = mappingVersionColumn()+"_FK10";
  description1.createForeignKey( fk10Name, mappingVersionColumn(),
                                 ora::OraMappingVersionTable::tableName(),ora::OraMappingVersionTable::mappingVersionColumn());
  std::string fk11Name = containerIdColumn()+"_FK11";
  description1.createForeignKey( fk11Name, containerIdColumn(),
                                 ora::OraContainerHeaderTable::tableName(),ora::OraContainerHeaderTable::containerIdColumn());
  m_schema.createTable( description1 ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
}

void ora::OraClassVersionTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

ora::OraMappingSchema::OraMappingSchema( coral::ISchema& dbSchema  ):
  m_schema( dbSchema ){
}

ora::OraMappingSchema::~OraMappingSchema(){
}

bool ora::OraMappingSchema::getVersionList( std::set<std::string>& dest ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraMappingVersionTable::tableName() ).newQuery() );
  query->addToOutputList( OraMappingVersionTable::mappingVersionColumn() );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string mappingVersion = currentRow[ OraMappingVersionTable::mappingVersionColumn()].data<std::string>();
    dest.insert( mappingVersion );
  }
  return ret;
}

bool ora::OraMappingSchema::getMapping( const std::string& version,
                                        ora::MappingRawData& dest ){
  bool ret = false;
  coral::ITable& mappingTable = m_schema.tableHandle( OraMappingElementTable::tableName() );
  std::auto_ptr<coral::IQuery> query(mappingTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<int>( OraMappingElementTable::elementIdColumn() );
  outputBuffer.extend<std::string>( OraMappingElementTable::elementTypeColumn() );
  outputBuffer.extend<std::string>( OraMappingElementTable::scopeNameColumn() );
  outputBuffer.extend<std::string>( OraMappingElementTable::variableNameColumn() );
  outputBuffer.extend<std::string>( OraMappingElementTable::variableTypeColumn() );
  outputBuffer.extend<std::string>( OraMappingElementTable::tableNameColumn() );
  outputBuffer.extend<std::string>( OraMappingElementTable::columnNameColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( OraMappingElementTable::elementIdColumn() );
  query->addToOutputList( OraMappingElementTable::elementTypeColumn() );
  query->addToOutputList( OraMappingElementTable::scopeNameColumn() );
  query->addToOutputList( OraMappingElementTable::variableNameColumn() );
  query->addToOutputList( OraMappingElementTable::variableTypeColumn() );
  query->addToOutputList( OraMappingElementTable::tableNameColumn() );
  query->addToOutputList( OraMappingElementTable::columnNameColumn() );
  std::ostringstream condition;
  condition << OraMappingElementTable::mappingVersionColumn()<<"= :"<< OraMappingElementTable::mappingVersionColumn();
  coral::AttributeList condData;
  condData.extend<std::string>( OraMappingElementTable::mappingVersionColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< std::string >() = version;
  query->setCondition( condition.str(), condData );
  query->addToOrderList( OraMappingElementTable::scopeNameColumn() );
  query->addToOrderList( OraMappingElementTable::variableNameColumn() );
  query->addToOrderList( OraMappingElementTable::variableParIndexColumn() );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    int elementId = currentRow[ OraMappingElementTable::elementIdColumn() ].data<int>();
    MappingRawElement& elem = dest.addElement( elementId );
    elem.elementType = currentRow[ OraMappingElementTable::elementTypeColumn() ].data<std::string>();
    elem.scopeName = currentRow[ OraMappingElementTable::scopeNameColumn() ].data<std::string>();
    elem.variableName = currentRow[ OraMappingElementTable::variableNameColumn() ].data<std::string>();
    elem.variableType = currentRow[ OraMappingElementTable::variableTypeColumn() ].data<std::string>();
    elem.tableName = currentRow[ OraMappingElementTable::tableNameColumn() ].data<std::string>();
    elem.columns.push_back( currentRow[ OraMappingElementTable::columnNameColumn() ].data<std::string>() );
  }
  return ret;
}

void ora::OraMappingSchema::storeMapping( const MappingRawData& mapping ){
  // first update the version table
  coral::ITable& mappingVersionTable = m_schema.tableHandle( OraMappingVersionTable::tableName() );
  coral::AttributeList  rowBuffer;
  rowBuffer.extend< std::string >( OraMappingVersionTable::mappingVersionColumn() );
  rowBuffer[ OraMappingVersionTable::mappingVersionColumn() ].data<std::string>()= mapping.version;
  mappingVersionTable.dataEditor().insertRow( rowBuffer );

  // then update the element tables
  coral::ITable& mappingElementTable = m_schema.tableHandle( OraMappingElementTable::tableName() );
  coral::AttributeList  dataBuffer;
  dataBuffer.extend< std::string >( OraMappingElementTable::mappingVersionColumn() );
  dataBuffer.extend< int >( OraMappingElementTable::elementIdColumn() );
  dataBuffer.extend< std::string >( OraMappingElementTable::elementTypeColumn() );
  dataBuffer.extend< std::string >( OraMappingElementTable::scopeNameColumn() );
  dataBuffer.extend< std::string >( OraMappingElementTable::variableNameColumn() );
  dataBuffer.extend< unsigned int >( OraMappingElementTable::variableParIndexColumn() );
  dataBuffer.extend< std::string >( OraMappingElementTable::variableTypeColumn() );
  dataBuffer.extend< std::string >( OraMappingElementTable::tableNameColumn() );
  dataBuffer.extend< std::string >( OraMappingElementTable::columnNameColumn() );
  dataBuffer[ OraMappingElementTable::mappingVersionColumn() ].data<std::string>()= mapping.version;

  for( std::map < int, MappingRawElement >::const_iterator iElem = mapping.elements.begin();
       iElem != mapping.elements.end(); iElem++ ){
    for( size_t iParamIndex = 0; iParamIndex < iElem->second.columns.size(); iParamIndex++ ){
      dataBuffer[ OraMappingElementTable::elementIdColumn() ].data<int>() = iElem->first;
      dataBuffer[ OraMappingElementTable::elementTypeColumn()].data<std::string>()=  iElem->second.elementType;
      dataBuffer[ OraMappingElementTable::scopeNameColumn() ].data<std::string>()= iElem->second.scopeName;
      dataBuffer[ OraMappingElementTable::variableNameColumn() ].data<std::string>()= iElem->second.variableName;
      dataBuffer[ OraMappingElementTable::variableParIndexColumn() ].data<unsigned int>() = iParamIndex;
      dataBuffer[ OraMappingElementTable::variableTypeColumn() ].data<std::string>()= iElem->second.variableType;
      dataBuffer[ OraMappingElementTable::tableNameColumn() ].data<std::string>()= iElem->second.tableName;
      dataBuffer[ OraMappingElementTable::columnNameColumn() ].data<std::string>()= iElem->second.columns[iParamIndex];
      mappingElementTable.dataEditor().insertRow( dataBuffer );
    }
  }
}

void ora::OraMappingSchema::removeMapping( const std::string& version ){
  // Remove all rows in the tables with the version.
  coral::AttributeList whereData;
  whereData.extend<std::string>( OraMappingVersionTable::mappingVersionColumn() );
  whereData.begin()->data<std::string>() = version;

  std::string condition = OraMappingVersionTable::mappingVersionColumn() + " = :" + OraMappingVersionTable::mappingVersionColumn();
  m_schema.tableHandle( OraClassVersionTable::tableName() ).dataEditor().deleteRows( condition, whereData );
  m_schema.tableHandle( OraMappingElementTable::tableName() ).dataEditor().deleteRows( condition, whereData );
  m_schema.tableHandle( OraMappingVersionTable::tableName() ).dataEditor().deleteRows( condition, whereData );
}

bool ora::OraMappingSchema::getContainerTableMap( std::map<std::string, int>& dest ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query(m_schema.newQuery());
  query->addToTableList( OraMappingElementTable::tableName(),"T0");
  query->addToTableList( OraClassVersionTable::tableName(), "T1");
  query->setDistinct();
  query->addToOutputList( "T0."+ OraMappingElementTable::tableNameColumn() );
  query->addToOutputList( "T1."+ OraClassVersionTable::containerIdColumn());
  std::ostringstream condition;
  condition << "T0."<<OraMappingElementTable::mappingVersionColumn()<<"="<< "T1."<< OraClassVersionTable::mappingVersionColumn();
  coral::AttributeList condData;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string tableName = currentRow[ "T0."+ OraMappingElementTable::tableNameColumn()].data<std::string>();
    int containerId = currentRow[ "T1."+ OraClassVersionTable::containerIdColumn()].data<int>();
    dest.insert(std::make_pair(tableName,containerId));
  }
  return ret;
}

/**bool ora::OraMappingSchema::getTableListForContainer( int containerId, std::set<std::string>& dest ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query(m_schema.newQuery());
  query->addToTableList( OraMappingElementTable::tableName(),"T0");
  query->addToTableList( OraMappingVersionTable::tableName(), "T1");
  query->setDistinct();
  query->addToOutputList( "T0."+ OraMappingElementTable::tableNameColumn() );
  std::ostringstream condition;
  condition << "T0."<<OraMappingElementTable::mappingVersionColumn()<<"="<< "T1."<< OraMappingVersionTable::mappingVersionColumn() << " AND ";
  condition << "T1."<<OraMappingVersionTable::containerIdColumn()<<" =:"<<OraMappingVersionTable::containerIdColumn();
  coral::AttributeList condData;
  condData.extend< int >( OraMappingVersionTable::containerIdColumn() );
  condData.begin()->data< int >() = containerId;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string tableName = currentRow[ "T0."+ OraMappingElementTable::tableNameColumn()].data<std::string>();
    dest.insert( tableName );
  }
  return ret;
}
**/

bool ora::OraMappingSchema::getDependentClassesInContainerMapping( int containerId,
                                                                   std::set<std::string>& destination ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraClassVersionTable::tableName() ).newQuery() );
  query->setDistinct();
  query->addToOutputList( OraClassVersionTable::classNameColumn() );
  std::ostringstream condition;
  condition <<OraClassVersionTable::containerIdColumn()<<" =:"<<OraClassVersionTable::containerIdColumn();
  condition << " AND "<< OraClassVersionTable::dependencyIndexColumn()<<" > 0";
  coral::AttributeList condData;
  condData.extend< int >( OraClassVersionTable::containerIdColumn() );
  condData[ OraClassVersionTable::containerIdColumn() ].data< int >() = containerId;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string className = currentRow[ OraClassVersionTable::classNameColumn() ].data<std::string>();
    destination.insert( className );
  }
  return ret;
}

bool ora::OraMappingSchema::getClassVersionListForMappingVersion( const std::string& mappingVersion,
                                                                  std::set<std::string>& destination ){
  
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraClassVersionTable::tableName() ).newQuery() );
  query->setDistinct();
  query->addToOutputList( OraClassVersionTable::classVersionColumn() );
  std::ostringstream condition;
  condition <<OraClassVersionTable::mappingVersionColumn()<<" =:"<<OraClassVersionTable::mappingVersionColumn();
  coral::AttributeList condData;
  condData.extend< std::string >( OraClassVersionTable::mappingVersionColumn() );  
  condData[ OraClassVersionTable::mappingVersionColumn() ].data< std::string >() = mappingVersion;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string classVersion = currentRow[ OraClassVersionTable::classVersionColumn() ].data<std::string>();
    destination.insert( classVersion );
  }
  return ret;
}

bool ora::OraMappingSchema::getMappingVersionListForContainer( int containerId,
                                                               std::set<std::string>& dest,
                                                               bool onlyDependency ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraClassVersionTable::tableName() ).newQuery() );
  query->setDistinct();
  query->addToOutputList( OraClassVersionTable::mappingVersionColumn() );
  std::ostringstream condition;
  condition <<OraClassVersionTable::containerIdColumn()<<" =:"<<OraClassVersionTable::containerIdColumn();
  coral::AttributeList condData;
  condData.extend< int >( OraClassVersionTable::containerIdColumn() );
  if( onlyDependency ){
    condition << " AND "<<OraClassVersionTable::dependencyIndexColumn()<<" > 0";
  }
  condData[ OraClassVersionTable::containerIdColumn() ].data< int >() = containerId;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string mappingVersion = currentRow[ OraClassVersionTable::mappingVersionColumn() ].data<std::string>();
    dest.insert( mappingVersion );
  }
  return ret;
}

bool ora::OraMappingSchema::getClassVersionListForContainer( int containerId,
                                                             std::map<std::string,std::string>& versionMap ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraClassVersionTable::tableName() ).newQuery() );
  query->setDistinct();
  query->addToOutputList( OraClassVersionTable::classVersionColumn() );
  query->addToOutputList( OraClassVersionTable::mappingVersionColumn() );
  std::ostringstream condition;
  condition <<OraClassVersionTable::containerIdColumn()<<" =:"<<OraClassVersionTable::containerIdColumn();
  coral::AttributeList condData;
  condData.extend< int >( OraClassVersionTable::containerIdColumn() );
  condData[ OraClassVersionTable::containerIdColumn() ].data< int >() = containerId;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string classVersion = currentRow[ OraClassVersionTable::classVersionColumn() ].data<std::string>();
    std::string mappingVersion = currentRow[ OraClassVersionTable::mappingVersionColumn() ].data<std::string>();
    versionMap.insert( std::make_pair(classVersion,mappingVersion ) );
  }
  return ret;
}

bool ora::OraMappingSchema::getMappingVersionListForTable( const std::string& tableName,
                                                           std::set<std::string>& destination )
{
  bool ret = false;
  destination.clear();
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraMappingElementTable::tableName() ).newQuery() );
  query->setDistinct();
  query->addToOutputList( OraMappingElementTable::mappingVersionColumn() );
  std::ostringstream condition;
  condition << OraMappingElementTable::tableNameColumn()<<" = :"<< OraMappingElementTable::tableNameColumn();
  coral::AttributeList condData;
  condData.extend< std::string >( OraMappingElementTable::tableNameColumn() );
  condData.begin()->data<std::string>() = tableName;
  query->setCondition(condition.str(),condData);
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    std::string mappingVersion = currentRow[ OraMappingElementTable::mappingVersionColumn()].data<std::string>();
    destination.insert( mappingVersion );
  }
  return ret;
}

bool ora::OraMappingSchema::selectMappingVersion( const std::string& classId,
                                                  int containerId,
                                                  std::string& destination ){
  bool ret = false;
  destination.clear();
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraClassVersionTable::tableName() ).newQuery() );
  query->addToOutputList( OraClassVersionTable::mappingVersionColumn() );
  std::ostringstream condition;
  condition << OraClassVersionTable::classIdColumn() << " =:" << OraClassVersionTable::classIdColumn() << " AND ";
  condition << OraClassVersionTable::containerIdColumn() << " =:" << OraClassVersionTable::containerIdColumn();
  coral::AttributeList condData;
  condData.extend<std::string>( OraClassVersionTable::classIdColumn() );
  condData.extend<int>( OraClassVersionTable::containerIdColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< std::string >() = classId;
  ++iAttribute;
  iAttribute->data< int >() = containerId;
  query->setCondition( condition.str(), condData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    destination = currentRow[OraClassVersionTable::mappingVersionColumn()].data<std::string>();
  }
  return ret;  
}

bool ora::OraMappingSchema::containerForMappingVersion( const std::string& mappingVersion,
                                                        int& destination ){
  bool ret = false;
  std::auto_ptr<coral::IQuery> query( m_schema.tableHandle( OraClassVersionTable::tableName() ).newQuery() );
  query->addToOutputList( OraClassVersionTable::containerIdColumn() );
  std::ostringstream condition;
  condition << OraClassVersionTable::mappingVersionColumn() << " =:"<< OraClassVersionTable::mappingVersionColumn();
  coral::AttributeList condData;
  condData.extend<std::string>( OraClassVersionTable::mappingVersionColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< std::string >() = mappingVersion;
  query->setCondition( condition.str(), condData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& currentRow = cursor.currentRow();
    destination = currentRow[ OraClassVersionTable::containerIdColumn() ].data<int>();
  }
  return ret;
}

void ora::OraMappingSchema::insertClassVersion( const std::string& className,
                                                const std::string& classVersion,
                                                const std::string& classId,
                                                int dependencyIndex,
                                                int containerId,
                                                const std::string& mappingVersion ){
  coral::ITable& classVersionTable = m_schema.tableHandle( OraClassVersionTable::tableName() );
  coral::AttributeList inputData;
  inputData.extend<std::string>( OraClassVersionTable::mappingVersionColumn());
  inputData.extend<std::string>( OraClassVersionTable::classNameColumn());
  inputData.extend<std::string>( OraClassVersionTable::classVersionColumn());
  inputData.extend<std::string>( OraClassVersionTable::classIdColumn());
  inputData.extend<int>( OraClassVersionTable::dependencyIndexColumn());
  inputData.extend<int>( OraClassVersionTable::containerIdColumn());
  coral::AttributeList::iterator iInAttr = inputData.begin();
  iInAttr->data< std::string >() = mappingVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = className;
  ++iInAttr;
  iInAttr->data< std::string >() = classVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = classId;
  ++iInAttr;
  iInAttr->data< int >() = dependencyIndex;
  ++iInAttr;
  iInAttr->data< int >() = containerId;
  classVersionTable.dataEditor().insertRow( inputData );
}

void ora::OraMappingSchema::setMappingVersion( const std::string& classId,
                                               int containerId,
                                               const std::string& mappingVersion ){
  coral::ITable& classVersionTable = m_schema.tableHandle( OraClassVersionTable::tableName() );
  coral::AttributeList inputData;
  inputData.extend<std::string>( OraClassVersionTable::mappingVersionColumn());
  inputData.extend<std::string>( OraClassVersionTable::classIdColumn());
  inputData.extend<int>( OraClassVersionTable::containerIdColumn());
  coral::AttributeList::iterator iInAttr = inputData.begin();
  iInAttr->data< std::string >() = mappingVersion;
  ++iInAttr;
  iInAttr->data< std::string >() = classId;
  ++iInAttr;
  iInAttr->data< int >() = containerId;
  std::string setClause = OraClassVersionTable::mappingVersionColumn()+" =:"+ OraClassVersionTable::mappingVersionColumn();
  std::string whereClause = OraClassVersionTable::classIdColumn()+" =:"+ OraClassVersionTable::classIdColumn()+" AND "+
    OraClassVersionTable::containerIdColumn()+" =:"+ OraClassVersionTable::containerIdColumn();
  classVersionTable.dataEditor().updateRows( setClause,whereClause, inputData  );
}

bool ora::OraDatabaseSchema::existsMainTable( coral::ISchema& dbSchema ){
  OraMainTable tmp( dbSchema );
  return tmp.exists();
}

std::string& ora::OraNamingServiceTable::tableName(){
  static std::string s_table("ORA_NAMING_SERVICE" );
  return s_table;
}

std::string& ora::OraNamingServiceTable::objectNameColumn(){
  static std::string s_column("OBJECT_NAME");
  return s_column;  
}

std::string& ora::OraNamingServiceTable::containerIdColumn(){
  static std::string s_column("CONTAINER_ID");
  return s_column;  
}

std::string& ora::OraNamingServiceTable::itemIdColumn(){
  static std::string s_column("ITEM_ID");
  return s_column;  
}

ora::OraNamingServiceTable::OraNamingServiceTable( coral::ISchema& dbSchema ): m_schema( dbSchema ){
}

ora::OraNamingServiceTable::~OraNamingServiceTable(){
}

bool ora::OraNamingServiceTable::exists(){
  return m_schema.existsTable( tableName() );
}

void ora::OraNamingServiceTable::create(){
  if( m_schema.existsTable( tableName() )){
    throwException( "ORA naming service table already exists in this schema.",
                    "OraNameTable::create");
  }
  
  coral::TableDescription descr( "OraDb" );
  descr.setName( tableName() );
  descr.insertColumn( objectNameColumn(),
                      coral::AttributeSpecification::typeNameForType<std::string>() );
  descr.insertColumn( containerIdColumn(),
                      coral::AttributeSpecification::typeNameForType<int>() );
  descr.insertColumn( itemIdColumn(),
                      coral::AttributeSpecification::typeNameForType<int>() );
  descr.setNotNullConstraint( objectNameColumn() );
  descr.setNotNullConstraint( containerIdColumn() );
  descr.setNotNullConstraint( itemIdColumn() );
  descr.setPrimaryKey( std::vector<std::string>( 1, objectNameColumn() ) );

  coral::ITable& table = m_schema.createTable( descr );
  table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
}

void ora::OraNamingServiceTable::drop(){
  m_schema.dropIfExistsTable( tableName() );
}

void ora::OraNamingServiceTable::setObjectName( const std::string& name, 
                                                int contId, 
                                                int itemId ){
  coral::AttributeList dataToInsert;
  dataToInsert.extend<std::string>( objectNameColumn() );
  dataToInsert.extend<int>( containerIdColumn());
  dataToInsert.extend<int>( itemIdColumn());
  dataToInsert[ objectNameColumn() ].data<std::string>() = name;
  dataToInsert[ containerIdColumn() ].data<int>()  = contId;
  dataToInsert[ itemIdColumn() ].data<int>()  = itemId;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  containerTable.dataEditor().insertRow( dataToInsert );  
}

bool ora::OraNamingServiceTable::eraseObjectName( const std::string& name ){
  coral::AttributeList whereData;
  whereData.extend<std::string>( objectNameColumn() );
  whereData.begin()->data<std::string>() = name;
  std::string condition = objectNameColumn() + " = :" + objectNameColumn();
  return m_schema.tableHandle( tableName() ).dataEditor().deleteRows( condition, whereData )>0;
}

bool ora::OraNamingServiceTable::eraseAllNames(){
  std::string condition("");
  coral::AttributeList whereData;
  return m_schema.tableHandle( tableName() ).dataEditor().deleteRows( condition, whereData )>0;
}

bool ora::OraNamingServiceTable::getObjectByName( const std::string& name, 
                                                  std::pair<int,int>& destination ){
  bool ret = false;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query( containerTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<int>( containerIdColumn() );
  outputBuffer.extend<int>( itemIdColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( containerIdColumn() );
  query->addToOutputList( itemIdColumn() );
  std::ostringstream condition;
  condition << objectNameColumn()<<"= :"<< objectNameColumn();
  coral::AttributeList condData;
  condData.extend<std::string>( objectNameColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< std::string >() = name;
  query->setCondition( condition.str(), condData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    int containerId = row[ containerIdColumn() ].data< int >();
    int itemId = row[ itemIdColumn() ].data< int >();
    destination.first = containerId;
    destination.second = itemId;
  }
  return ret;
}

bool ora::OraNamingServiceTable::getNamesForObject( int contId, 
                                                    int itemId, 
                                                    std::vector<std::string>& destination ){
  bool ret = false;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query( containerTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<std::string>( objectNameColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( objectNameColumn() );
  std::ostringstream condition;
  condition << containerIdColumn()<<"= :"<< containerIdColumn();
  condition << " AND ";
  condition << itemIdColumn()<<"= :"<< itemIdColumn();
  coral::AttributeList condData;
  condData.extend<int>( containerIdColumn() );
  condData.extend<int>( itemIdColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< int >() = contId;
  ++iAttribute;
  iAttribute->data< int >() = itemId;
  query->setCondition( condition.str(), condData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    std::string name = row[ objectNameColumn() ].data< std::string >();
    destination.push_back( name );
  }
  return ret;
}

bool ora::OraNamingServiceTable::getNamesForContainer( int contId, 
                                                       std::vector<std::string>& destination ){
  bool ret = false;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query( containerTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<std::string>( objectNameColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( objectNameColumn() );
  std::ostringstream condition;
  condition << containerIdColumn()<<"= :"<< containerIdColumn();
  coral::AttributeList condData;
  condData.extend<int>( containerIdColumn() );
  coral::AttributeList::iterator iAttribute = condData.begin();
  iAttribute->data< int >() = contId;
  query->setCondition( condition.str(), condData );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    std::string name = row[ objectNameColumn() ].data< std::string >();
    destination.push_back( name );
  }
  return ret;
}

bool ora::OraNamingServiceTable::getAllNames( std::vector<std::string>& destination ){
  bool ret = false;
  coral::ITable& containerTable = m_schema.tableHandle( tableName() );
  std::auto_ptr<coral::IQuery> query( containerTable.newQuery());
  coral::AttributeList outputBuffer;
  outputBuffer.extend<std::string>( objectNameColumn() );
  query->defineOutput( outputBuffer );
  query->addToOutputList( objectNameColumn() );
  coral::ICursor& cursor = query->execute();
  while ( cursor.next() ) {
    ret = true;
    const coral::AttributeList& row = cursor.currentRow();
    std::string name = row[ objectNameColumn() ].data< std::string >();
    destination.push_back( name );
  }
  return ret;
}

ora::OraDatabaseSchema::OraDatabaseSchema( coral::ISchema& dbSchema ):
  IDatabaseSchema( dbSchema ),
  m_schema( dbSchema ),
  m_mainTable( dbSchema ),
  m_sequenceTable( dbSchema ),
  m_mappingVersionTable( dbSchema ),
  m_mappingElementTable( dbSchema ),
  m_containerHeaderTable( dbSchema ),
  m_classVersionTable( dbSchema ),
  m_mappingSchema( dbSchema ),
  m_namingServiceTable( dbSchema ){
}

ora::OraDatabaseSchema::~OraDatabaseSchema(){
}

bool ora::OraDatabaseSchema::exists(){
  if(!m_mainTable.exists()){
    return false;
  }
  if(!m_sequenceTable.exists() ||
     !m_mappingVersionTable.exists() ||
     !m_mappingElementTable.exists() ||
     !m_containerHeaderTable.exists() ||
     !m_classVersionTable.exists() || 
     !m_namingServiceTable.exists()){
    throwException( "ORA database is corrupted..",
                    "OraDatabaseSchema::exists");
  }
  return true;
}

void ora::OraDatabaseSchema::create( const std::string& userSchemaVersion ){
  m_mainTable.create();
  m_mainTable.setParameter( IMainTable::userSchemaVersionParameterName(), userSchemaVersion );
  m_sequenceTable.create();
  m_mappingVersionTable.create();
  m_mappingElementTable.create();
  m_containerHeaderTable.create();
  m_classVersionTable.create();
  m_namingServiceTable.create();
}

void ora::OraDatabaseSchema::drop(){
  m_namingServiceTable.drop();
  m_classVersionTable.drop();
  m_containerHeaderTable.drop();
  m_mappingElementTable.drop();
  m_mappingVersionTable.drop();
  m_sequenceTable.drop();
  m_mainTable.drop(); 
}

ora::IMainTable& ora::OraDatabaseSchema::mainTable(){
  return m_mainTable;
}

ora::ISequenceTable& ora::OraDatabaseSchema::sequenceTable(){
  return m_sequenceTable;
}

ora::IDatabaseTable& ora::OraDatabaseSchema::mappingVersionTable(){
  return m_mappingVersionTable;  
}

ora::IDatabaseTable& ora::OraDatabaseSchema::mappingElementTable(){
  return m_mappingElementTable;  
}

ora::IContainerHeaderTable& ora::OraDatabaseSchema::containerHeaderTable(){
  return m_containerHeaderTable;
}

ora::IDatabaseTable& ora::OraDatabaseSchema::classVersionTable(){
  return m_classVersionTable;  
}

ora::IMappingSchema& ora::OraDatabaseSchema::mappingSchema(){
  return m_mappingSchema;  
}

ora::INamingServiceTable& ora::OraDatabaseSchema::namingServiceTable(){
  return m_namingServiceTable;
}
  
    
