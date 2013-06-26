#include "CondTools/DQM/interface/TestBase.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/IConnection.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/RelationalServiceException.h"
#include "CoralKernel/Context.h"
#include<iostream>


TestBase::TestBase():
  m_connection( 0 ) {
  coral::Context& context = coral::Context::instance();
  context.loadComponent( "CORAL/RelationalPlugins/oracle" );
  coral::IHandle<coral::IRelationalDomain> domain = context.query<coral::IRelationalDomain>( "CORAL/RelationalPlugins/oracle" );    
  if ( ! domain.isValid() )
    throw std::runtime_error( "Could not load the OracleAccess plugin" );
}

TestBase::~TestBase() {
  if ( m_connection ) delete m_connection;
}

coral::ISession*
TestBase::connect( const std::string& connectionString,
                   const std::string& user,
                   const std::string& pass ) {
  coral::Context& ctx = coral::Context::instance();
  coral::IHandle<coral::IRelationalDomain> iHandle=ctx.query<coral::IRelationalDomain>("CORAL/RelationalPlugins/oracle");
  if ( ! iHandle.isValid() ) 
    throw std::runtime_error( "Could not load the OracleAccess plugin" );
  
  std::pair<std::string, std::string> connectionAndSchema = iHandle->decodeUserConnectionString( connectionString );
  if ( ! m_connection )
    m_connection = iHandle->newConnection( connectionAndSchema.first );

  if ( ! m_connection->isConnected() )
    m_connection->connect();

  coral::ISession* session = m_connection->newSession( connectionAndSchema.second );
  if ( session ) 
    session->startUserSession( user, pass );
  
  return session;
}

void
TestBase::setVerbosityLevel( coral::MsgLevel level ) {
  coral::MessageStream::setMsgVerbosity(level);
}
