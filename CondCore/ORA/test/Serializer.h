#ifndef INCLUDE_ORA_SERIALIZER_H
#define INCLUDE_ORA_SERIALIZER_H

#include "CondCore/ORA/interface/Exception.h"
//
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITablePrivilegeManager.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
//

namespace ora {

  class Serializer {
    public:
    explicit Serializer( const std::string& name ):
      m_tableName( name+"_LOCK" ),
      m_connServ(),
      m_session(),
      m_lock( false ){
    }
      
    virtual ~Serializer(){
      release();
    }
    
    void init( const std::string& connectionString ){
      coral::ISchema& schema = m_session->nominalSchema();
      if(!schema.existsTable( m_tableName )){
        coral::TableDescription descr( "OraDb" );
        descr.setName( m_tableName );
        descr.insertColumn( "LOCK",
                            coral::AttributeSpecification::typeNameForType<int>() );
        descr.setNotNullConstraint( "LOCK" );
        descr.setPrimaryKey( std::vector<std::string>( 1, "LOCK" ) );
        coral::ITable& table = schema.createTable( descr );
        table.privilegeManager().grantToPublic( coral::ITablePrivilegeManager::Select );
      }
    }

    void lock( const std::string& connectionString ){
      if( !m_lock ){
        m_connServ.configuration().setConnectionTimeOut(0);
        m_session.reset( m_connServ.connect( connectionString, coral::Update ) );
        m_session->transaction().start( false );
        init( connectionString );
        coral::ISchema& schema = m_session->nominalSchema();
        coral::ITable& table = schema.tableHandle( m_tableName );
        std::string condition("LOCK = 1");
        std::auto_ptr<coral::IQuery> query( table.newQuery() );
        query->addToOutputList( "LOCK" );
        query->defineOutputType( "LOCK", coral::AttributeSpecification::typeNameForType<int>());
        query->setCondition( condition, coral::AttributeList() );
        query->setForUpdate();
        coral::ICursor& cursor = query->execute();
        coral::AttributeList data;
        data.extend<int>( "LOCK" );
        data["LOCK"].data<int>() = 1;
        if( cursor.next() ){
          // row found. will be locked by the DB if some other session owns the transaction...
          std::string setCLause = "LOCK = :LOCK";
          table.dataEditor().updateRows( setCLause, condition , data );
        } else {
          // no row found... no lock!
          table.dataEditor().insertRow( data );
        }
        m_lock = true;
      }
    }
    

    void release(){
      if( m_lock ){
        m_lock = false;
        m_session->transaction().commit();
      }
    }
    

    private:

    std::string m_tableName;
    coral::ConnectionService m_connServ;
    std::auto_ptr<coral::ISessionProxy> m_session;
    bool m_lock;
    
  };
  
}

#endif


