#include "CondCore/ORA/interface/SchemaUtils.h"
// externals
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITableSchemaEditor.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IForeignKey.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
//
#include <memory>

void ora::SchemaUtils::cleanUp( const std::string& connectionString, std::set<std::string> exclusionList ){
  coral::ConnectionService connServ;
  std::auto_ptr<coral::ISessionProxy> session( connServ.connect( connectionString, coral::Update ));
  session->transaction().start();
  try{
    coral::ISchema& schema = session->nominalSchema();
    std::set<std::string> tables = schema.listTables();
    for( std::set<std::string>::const_iterator iEx = exclusionList.begin();
	 iEx != exclusionList.end(); ++iEx ){
      tables.erase( *iEx );
    }
    for( std::set<std::string>::const_iterator iT = tables.begin();
	 iT != tables.end(); ++iT ){
      coral::ITable& t = schema.tableHandle( *iT );
      int numFKs = t.description().numberOfForeignKeys();
      for( int ifk=0; ifk < numFKs; ifk++ ){
        // workaround: since the dropFK triggers a commit, the fk list is reset. therefore, always drop the fk id=0!!!
	t.schemaEditor().dropForeignKey( t.description().foreignKey( 0 ).name() );
      };
    }
    for( std::set<std::string>::const_iterator iT = tables.begin();
	 iT != tables.end(); ++iT ){
      schema.dropIfExistsTable( *iT );
    }
    session->transaction().commit();
  } catch ( ... ){
    session->transaction().rollback();
    throw;
  }
}  

const std::string& ora::Serializer::tableName(){
  static const std::string s_tableName("ORA_LOCK");
  return s_tableName;
}

ora::Serializer::Serializer():
  m_connServ( new coral::ConnectionService ),
  m_session(),
  m_lock( false ){
}
      
ora::Serializer::~Serializer(){
  release();
}
    
void ora::Serializer::lock( const std::string& connectionString ){
  if( !m_lock ){
    m_connServ->configuration().setConnectionTimeOut(0);
    m_session.reset( m_connServ->connect( connectionString, coral::Update ) );
    m_session->transaction().start( false );
    coral::ISchema& schema = m_session->nominalSchema();
    if(!schema.existsTable( tableName() )){
      coral::TableDescription descr( "OraDb" );
      descr.setName( tableName() );
      descr.insertColumn( "P_LOCK",
			  coral::AttributeSpecification::typeNameForType<int>() );
      descr.setNotNullConstraint( "P_LOCK" );
      descr.setPrimaryKey( std::vector<std::string>( 1, "P_LOCK" ) );
      coral::ITable& table = schema.createTable( descr );
      table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
    }
    coral::ITable& table = schema.tableHandle( tableName() );
    std::string condition("P_LOCK = 1");
    std::auto_ptr<coral::IQuery> query( table.newQuery() );
    query->addToOutputList( "P_LOCK" );
    query->defineOutputType( "P_LOCK", coral::AttributeSpecification::typeNameForType<int>());
    query->setCondition( condition, coral::AttributeList() );
    query->setForUpdate();
    coral::ICursor& cursor = query->execute();
    coral::AttributeList data;
    data.extend<int>( "P_LOCK" );
    data["P_LOCK"].data<int>() = 1;
    if( cursor.next() ){
      // row found. will be locked by the DB if some other session owns the transaction...
      std::string setCLause = "P_LOCK = :P_LOCK";
      table.dataEditor().updateRows( setCLause, condition , data );
    } else {
      // no row found... no lock!
      table.dataEditor().insertRow( data );
    }
    m_lock = true;
  }
}
    

void ora::Serializer::release(){
  if( m_lock ){
    m_lock = false;
    m_session->transaction().commit();
  }
}
    

