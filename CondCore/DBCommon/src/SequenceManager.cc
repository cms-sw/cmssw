#include "CondCore/DBCommon/interface/SequenceManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"
#include "RelationalAccess/SchemaException.h"
#include <memory>
cond::SequenceManager::SequenceManager(cond::DbSession& coraldb,
				       const std::string& sequenceTableName):
  m_coraldb(coraldb),
  m_sequenceTableName( sequenceTableName ),
  m_tableToId(),
  m_sequenceTableExists( false ),
  m_whereClause( std::string("REFTABLE_NAME")+" =:"+std::string("REFTABLE_NAME")), 
  m_whereData( 0 ),
  m_setClause( std::string("IDVALUE")+" = "+std::string("IDVALUE")+" + 1"),
  m_started( false )
{
  m_whereData = new coral::AttributeList; 
  m_whereData->extend<std::string>(std::string("REFTABLE_NAME"));
  init();
}
void
cond::SequenceManager::init(){ 
  m_sequenceTableExists=m_coraldb.nominalSchema().existsTable(m_sequenceTableName) ;
  m_started=true;
}
cond::SequenceManager::~SequenceManager()
{  
  delete m_whereData;
}
unsigned long long
cond::SequenceManager::incrementId( const std::string& tableName ){
  std::map< std::string, unsigned long long >::iterator iSequence = m_tableToId.find( tableName );
  coral::ISchema& schema=m_coraldb.nominalSchema();
  if ( iSequence == m_tableToId.end() ) {
    // Make sure that the sequence table exists.
    if ( ! m_sequenceTableExists ) {
      throw cond::Exception("SequenceManager::incrementId");
    }
    // Lock the entry in the table.
    unsigned long long lastIdUsed = 0;
    if ( ! ( this->lockEntry( schema, tableName, lastIdUsed ) ) ) {
      // Create the entry in the table if it does not exist.
      coral::AttributeList rowData;
      rowData.extend<std::string>("REFTABLE_NAME");
      rowData.extend<unsigned long long>("IDVALUE");
      coral::AttributeList::iterator iAttribute = rowData.begin();
      iAttribute->data< std::string >() = tableName;
      ++iAttribute;
      unsigned long long startingIdValue = lastIdUsed;
      iAttribute->data< unsigned long long >() = startingIdValue;
      try{
	schema.tableHandle( m_sequenceTableName ).dataEditor().insertRow( rowData );
	m_tableToId.insert( std::make_pair( tableName, startingIdValue ) );
	return startingIdValue;
      }catch(const coral::DataEditorException& er){
	this->lockEntry( schema, tableName, lastIdUsed );
	++lastIdUsed;
	iSequence = m_tableToId.insert( std::make_pair( tableName, lastIdUsed ) ).first;
	m_whereData->begin()->data<std::string>() = tableName;
	schema.tableHandle(m_sequenceTableName).dataEditor().updateRows(m_setClause,m_whereClause,*m_whereData );
	return lastIdUsed;
	//startingIdValue = lastIdUsed+1;
	//m_tableToId.insert( std::make_pair( tableName, startingIdValue ) );
      }catch(std::exception& er){
	throw cond::Exception(er.what());
      }

    }
    // Add the entry into the map.
    iSequence = m_tableToId.insert( std::make_pair( tableName, lastIdUsed ) ).first;
  }
  // Increment the oid transiently
  unsigned long long& oid = iSequence->second;
  this->lockEntry( schema, tableName, oid );
  ++oid;
  // Increment the oid in the database as well
  
  m_whereData->begin()->data<std::string>() = tableName;
  schema.tableHandle(m_sequenceTableName).dataEditor().updateRows(m_setClause,m_whereClause,*m_whereData );
  
  return oid;
}
/*
void
cond::SequenceManager::updateId( const std::string& tableName, 
				 unsigned long long lastId ){
  // Make sure that the sequence table exists.
  if ( ! m_sequenceTableExists ) {
    throw;
  }
  bool update = false;
  coral::IQuery* query = m_coraldb.nominalSchema().tableHandle( m_sequenceTableName ).newQuery();
  query->limitReturnedRows( 1, 0 );
  query->addToOutputList( std::string("IDVALUE") );
  query->defineOutputType( std::string("IDVALUE"),coral::AttributeSpecification::typeNameForType<unsigned long long>() );
  m_whereData->begin()->data< std::string >() = tableName;
  query->setCondition( m_whereClause, *m_whereData );
  coral::ICursor& cursor = query->execute();
  if ( cursor.next() ) {
    update = true;
  }
  delete query;

  coral::AttributeList rowData;
  rowData.extend<unsigned long long>( std::string("IDVALUE") );
  rowData.extend<std::string>(  std::string("REFTABLE_NAME") );
  coral::AttributeList::iterator iAttribute = rowData.begin();
  iAttribute->data< unsigned long long >() = lastId;
  ++iAttribute;
  iAttribute->data< std::string >() = tableName;
  coral::ISchema& schema= m_coraldb.nominalSchema();
  if ( update ) {
    // Update the entry in the table
    std::string setClause(std::string("IDVALUE")+" =: "+std::string("IDVALUE"));
    schema.tableHandle( m_sequenceTableName ).dataEditor().updateRows( setClause,m_whereClause,rowData );
    m_tableToId.erase( tableName );
  } else {
    schema.tableHandle( m_sequenceTableName ).dataEditor().insertRow( rowData );
  }
  m_tableToId.insert( std::make_pair( tableName, lastId ) );
}
*/
void
cond::SequenceManager::clear()
{
  m_tableToId.clear();
}
bool
cond::SequenceManager::existSequencesTable(){
  return m_sequenceTableExists;
}
void
cond::SequenceManager::createSequencesTable()
{
  coral::ISchema& schema= m_coraldb.nominalSchema();
  coral::TableDescription description( "CONDSEQ" );
  description.setName(m_sequenceTableName);
  description.insertColumn(std::string("REFTABLE_NAME"),
			   coral::AttributeSpecification::typeNameForType<std::string>() );
  description.setNotNullConstraint(std::string("REFTABLE_NAME"));
  description.insertColumn(std::string("IDVALUE"),
			   coral::AttributeSpecification::typeNameForType<unsigned long long>() );
  description.setNotNullConstraint(std::string("IDVALUE"));
  description.setPrimaryKey( std::vector< std::string >( 1, std::string("REFTABLE_NAME")));
  schema.createTable( description ).privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
  m_sequenceTableExists=true;
}

bool
cond::SequenceManager::lockEntry( coral::ISchema& schema,
				  const std::string& sequenceName,
				  unsigned long long& lastId ){
  std::auto_ptr< coral::IQuery > query( schema.tableHandle(m_sequenceTableName).newQuery());
  query->limitReturnedRows( 1, 0 );
  query->addToOutputList( std::string("IDVALUE") );
  query->defineOutputType( std::string("IDVALUE"), coral::AttributeSpecification::typeNameForType<unsigned long long>() );
  query->setForUpdate();
  m_whereData->begin()->data< std::string >() = sequenceName;
  query->setCondition( m_whereClause, *m_whereData );
  coral::ICursor& cursor = query->execute();
  if ( cursor.next() ) {
    lastId = cursor.currentRow().begin()->data< unsigned long long >();
    return true;
  }else {
    cursor.close();
    return false;
  }
}
