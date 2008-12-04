#include "TestBase.h"
#include "RelationalAccess/IRelationalDomain.h"
#include "RelationalAccess/IConnection.h"
#include "RelationalAccess/ISession.h"
#include "RelationalAccess/RelationalServiceException.h"
#include "PluginManager/PluginManager.h"
#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"

TestBase::TestBase():
  m_context( new seal::Context ),
  m_connection( 0 )
{
  seal::PluginManager* pm = seal::PluginManager::get();
  pm->initialise();
  seal::Handle<seal::ComponentLoader> loader = new seal::ComponentLoader( m_context.get() );
  if ( ! loader->load( "CORAL/RelationalPlugins/oracle" ) ) {
    throw std::runtime_error( "Could not load the OracleAccess plugin" );
  }
}


TestBase::~TestBase()
{
  if ( m_connection ) delete m_connection;
  m_context = 0;
}


coral::ISession*
TestBase::connect( const std::string& connectionString,
                   const std::string& userName,
                   const std::string& password )
{
  seal::IHandle<coral::IRelationalDomain> iHandle = m_context->query<coral::IRelationalDomain>( "CORAL/RelationalPlugins/oracle" );
  if ( ! iHandle ) {
    throw coral::NonExistingDomainException( "oracle" );
  }

  std::pair<std::string, std::string> connectionAndSchema = iHandle->decodeUserConnectionString( connectionString );

  if ( ! m_connection )
    m_connection = iHandle->newConnection( connectionAndSchema.first );

  if ( ! m_connection->isConnected() )
    m_connection->connect();

  coral::ISession* session = m_connection->newSession( connectionAndSchema.second );

  if ( session ) {
    session->startUserSession( userName, password );
  }

  return session;
}


void
TestBase::setVerbosityLevel( seal::Msg::Level level )
{
  std::vector< seal::Handle<seal::IMessageService> > v_msgSvc;
  m_context->query( v_msgSvc );
  if ( v_msgSvc.empty() ) {
    seal::Handle<seal::ComponentLoader> loader = new seal::ComponentLoader( m_context.get() );
    if ( ! loader->load( "SEAL/Services/MessageService" ) ) {
      throw std::runtime_error( "Could not load the seal message service" );
    }
    
    m_context->query( v_msgSvc );
    if ( v_msgSvc.empty() ) {
      throw std::runtime_error( "Could not load the seal message service" );
    }
  }
  seal::Handle<seal::IMessageService>& msgSvc = v_msgSvc.front();
  msgSvc->setOutputLevel( level );
}
